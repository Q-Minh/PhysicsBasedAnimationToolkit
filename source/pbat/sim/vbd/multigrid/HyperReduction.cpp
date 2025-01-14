#include "HyperReduction.h"

#include "pbat/common/Eigen.h"
#include "pbat/common/Indexing.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/graph/Partition.h"
#include "pbat/math/PolynomialBasis.h"
#include "pbat/math/SymmetricQuadratureRules.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"
#include "pbat/sim/vbd/multigrid/Hierarchy.h"

#include <ranges>
#include <tbb/parallel_for.h>
#include <utility>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

HyperReduction::HyperReduction(Hierarchy const& hierarchy, Index clusterSize)
    : Cptr(hierarchy.levels.size()),
      Cadj(hierarchy.levels.size()),
      eC(hierarchy.levels.size()),
      ApInvC(hierarchy.levels.size()),
      Ep(hierarchy.levels.size() + 1),
      EpMax(1e-6)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.Construct");
    ConstructHierarchicalClustering(hierarchy, clusterSize);
    auto nElements = hierarchy.data.E.cols();
    auto nLevels   = hierarchy.levels.size();
    AllocateWorkspace(nElements, nLevels);
    SelectClusterRepresentatives(hierarchy);
    PrecomputeInversePolynomialMatrices(hierarchy);
}

void HyperReduction::AllocateWorkspace(Index nElements, std::size_t nLevels)
{
    Ep.resize(nLevels + 1);
    Ep.front().resize(nElements);
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const nClusters = Cptr[l].size() - 1;
        eC[l].resize(nClusters);
        ApInvC[l].resize(4, 4 * nClusters);
        Ep[l + 1].resize(nClusters);
    }
}

void HyperReduction::ConstructHierarchicalClustering(Hierarchy const& hierarchy, Index clusterSize)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.vbd.multigrid.HyperReduction.ConstructHierarchicalClustering");

    // Construct the graph of face-adjacent fine mesh elements
    IndexVectorX Gptr, Gadj, Gwts;
    std::tie(Gptr, Gadj, Gwts) = graph::MatrixToWeightedAdjacency(graph::MeshDualGraph(
        hierarchy.data.E,
        hierarchy.data.X.cols(),
        graph::EMeshDualGraphOptions::FaceAdjacent));

    // Ask for contiguous partitions
    graph::PartitioningOptions opts{};
    opts.eCoarseningStrategy =
        graph::PartitioningOptions::ECoarseningStrategy::SortedHeavyEdgeMatching;
    opts.rngSeed                        = 0;
    opts.bMinimizeSupernodalGraphDegree = true;
    opts.bEnforceContiguousPartitions   = true;
    opts.bIdentifyConnectedComponents   = true;

    // Construct the clustering at each level
    auto nLevels = hierarchy.levels.size();
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        // Reduce the graph via partitioning/clustering
        auto nGraphNodes = Gptr.size() - 1;
        auto nPartitions = nGraphNodes / clusterSize;
        auto clustering  = graph::Partition(Gptr, Gadj, Gwts, nPartitions, opts);
        // Store the l^{th} level clustering
        std::tie(Cptr[l], Cadj[l]) = graph::MapToAdjacency(clustering);
        // Exit if coarsest level reached
        if (l + 1 == nLevels)
            break;
        // Compute the supernodal graph (i.e. graph of clusters) as next graph to partition
        auto Gsizes = Gptr(Eigen::seqN(1, nGraphNodes)) - Gptr(Eigen::seqN(0, nGraphNodes));
        auto u      = common::Repeat(
            IndexVectorX::LinSpaced(clustering.size(), 0, clustering.size() - 1),
            Gsizes);
        auto const& v = Gadj;
        auto SGu      = clustering(u);
        auto SGv      = clustering(v);
        auto edgeView = std::views::iota(0, SGu.size()) | std::views::transform([&](auto i) {
                            return graph::WeightedEdge(SGu(i), SGv(i), Gwts(i));
                        }) |
                        std::views::common;
        // NOTE:
        // Unfortunately, Eigen does not support std::ranges iterators, because it expects iterators
        // to have operator->. We have to make a copy into a std::vector to use Eigen's sparse
        // matrix.
        std::vector<graph::WeightedEdge<Index, Index>> edges(edgeView.begin(), edgeView.end());
        std::tie(Gptr, Gadj, Gwts) = graph::MatrixToWeightedAdjacency(
            graph::AdjacencyMatrixFromEdges(edges.begin(), edges.end()));
    }
}

void HyperReduction::SelectClusterRepresentatives(Hierarchy const& hierarchy)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.SelectClusterRepresentatives");

    auto const& data     = hierarchy.data;
    auto const& E        = data.E;
    auto const& X        = data.X;
    auto const nElements = E.cols();
    auto const dims      = X.rows();
    MatrixX centroids(dims, nElements);

    // Compute element centroids
    tbb::parallel_for(Index(0), nElements, [&](Index e) {
        auto const& x    = X(Eigen::placeholders::all, E.col(e));
        centroids.col(e) = x.rowwise().mean();
    });

    // Select each level's cluster representatives
    auto nLevels = hierarchy.levels.size();
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const& cptr     = Cptr[l];
        auto const& cadj     = Cadj[l];
        auto const nClusters = cptr.size() - 1;
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto const& cluster = cadj(Eigen::seq(cptr(c), cptr(c + 1) - 1));
            auto const& Xc      = centroids(Eigen::placeholders::all, cluster);
            auto const muXc     = Xc.rowwise().mean().eval();
            auto const d2mu     = (Xc.array().colwise() - muXc.array()).colwise().squaredNorm();
            Index eMu;
            [[maybe_unused]] Scalar d2eMu = d2mu.minCoeff(&eMu);
            eC[l](c)                      = cluster(eMu);
            centroids.col(c)              = muXc;
        });
    }
}

void HyperReduction::PrecomputeInversePolynomialMatrices(Hierarchy const& hierarchy)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.vbd.multigrid.HyperReduction.PrecomputeInversePolynomialMatrices");

    auto const& data = hierarchy.data;
    auto const& E    = data.E;
    auto const& detJe =
        data.wg; // wg = detJe*wge, but with a single wge per tetrahedron, we have wge=1
    auto nFineElements = E.cols();
    MatrixX Ap(4, 4 * nFineElements);
    Ap.setZero();

    // Compute the element polynomial inner product matrices
    using Polynomial = math::MonomialBasis<kDims, kPolynomialOrder>;
    using Quadrature = math::SymmetricSimplexPolynomialQuadratureRule<kDims, 2 * kPolynomialOrder>;
    tbb::parallel_for(Index(0), nFineElements, [&](Index e) {
        auto const& Xe = data.X(Eigen::placeholders::all, E.col(e));
        auto Xig       = common::ToEigen(Quadrature::points)
                       .reshaped(Quadrature::kDims + 1, Quadrature::kPoints);
        auto wg  = common::ToEigen(Quadrature::weights);
        auto Ape = Ap.block<4, 4>(0, 4 * e);
        for (auto g = 0; g < Quadrature::kPoints; ++g)
        {
            auto Xg = Xe * Xig.col(g);
            auto P  = Polynomial{}.eval(Xg);
            Ape += wg(g) * P * P.transpose() * detJe(e);
        }
    });

    // Compute the inverse of the per cluster polynomial inner product matrices
    auto nLevels = hierarchy.levels.size();
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const& cptr     = Cptr[l];
        auto const& cadj     = Cadj[l];
        auto const nClusters = cptr.size() - 1;
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto const& cluster = cadj(Eigen::seq(cptr(c), cptr(c + 1) - 1));
            auto Apc            = Ap.block<4, 4>(0, 4 * cluster(0));
            for (auto cc = 1; cc < cluster.size(); ++cc)
                Apc += Ap.block<4, 4>(0, 4 * cluster(cc));
            auto ApInvc = ApInvC[l].block<4, 4>(0, 4 * c);
            ApInvc      = Apc.inverse();
        });
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat
