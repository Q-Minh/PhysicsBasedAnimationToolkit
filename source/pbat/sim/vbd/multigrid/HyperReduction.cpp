#include "HyperReduction.h"

#include "pbat/common/Indexing.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/graph/Partition.h"
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
      PinvC(hierarchy.levels.size()),
      Ep(hierarchy.levels.size() + 1),
      EpMax(1e-6)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.Construct");
    ConstructHierarchicalClustering(hierarchy, clusterSize);
    SelectClusterRepresentatives(hierarchy);
    PrecomputeInversePolynomialMatrices(hierarchy);
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
    for (decltype(nLevels) l = 0; l < nLevels; l++)
    {
        // Reduce the graph via partitioning/clustering
        auto nGraphNodes = Gptr.size() - 1;
        auto nPartitions = nGraphNodes / clusterSize;
        auto clustering  = graph::Partition(Gptr, Gadj, Gwts, nPartitions, opts);
        // Store the l^{th} level clustering
        std::tie(Cptr[l], Cadj[l]) = graph::MapToAdjacency(clustering);
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
        auto const& x    = X(E.col(e));
        centroids.col(e) = x.rowwise().mean();
    });

    // Select each level's cluster representatives
    auto nLevels = hierarchy.levels.size();
    for (decltype(nLevels) l = 0; l < nLevels; l++)
    {
        auto const& cptr     = Cptr[l];
        auto const& cadj     = Cadj[l];
        auto& ec             = eC[l];
        auto const nClusters = cptr.size() - 1;
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto const& cluster = cadj(Eigen::seqN(cptr(c), cptr(c + 1)));
            auto const& Xc      = centroids(Eigen::placeholders::all, cluster);
            auto const muXc     = Xc.rowwise().mean().eval();
            auto const d2mu     = (Xc.array().colwise() - muXc.array()).colwise().squaredNorm();
            Index eMu;
            Scalar d2eMu     = d2mu.minCoeff(&eMu);
            ec(c)            = cluster(eMu);
            centroids.col(c) = muXc;
        });
    }
}

void HyperReduction::PrecomputeInversePolynomialMatrices(Hierarchy const& hierarchy)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.vbd.multigrid.HyperReduction.PrecomputeInversePolynomialMatrices");

    auto nLevels = hierarchy.levels.size();
    for (decltype(nLevels) l = 0; l < nLevels; l++)
    {
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat
