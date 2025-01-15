#include "HyperReduction.h"

#include "pbat/common/Eigen.h"
#include "pbat/common/Indexing.h"
#include "pbat/fem/ShapeFunctions.h"
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

HyperReduction::HyperReduction(Hierarchy const& hierarchy, Index clusterSize) : HyperReduction()
{
    Construct(hierarchy, clusterSize);
}

void HyperReduction::Construct(Hierarchy const& hierarchy, Index clusterSize)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.Construct");
    ConstructHierarchicalClustering(hierarchy, clusterSize);
    auto nLevels = hierarchy.levels.size();
    AllocateWorkspace(nLevels);
    SelectClusterRepresentatives(hierarchy);
    PrecomputeInversePolynomialMatrices(hierarchy);
}

void HyperReduction::AllocateWorkspace(std::size_t nLevels)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.AllocateWorkspace");
    eC.resize(nLevels);
    ApInvC.resize(nLevels);
    up.resize(nLevels);
    Ep.resize(nLevels);
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const nClusters = Cptr[l].size() - 1;
        eC[l].resize(nClusters);
        ApInvC[l].resize(4, 4 * nClusters);
        Ep[l].resize(nClusters);
    }
}

void HyperReduction::ConstructHierarchicalClustering(Hierarchy const& hierarchy, Index clusterSize)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.vbd.multigrid.HyperReduction.ConstructHierarchicalClustering");

    Cptr.resize(hierarchy.levels.size());
    Cadj.resize(hierarchy.levels.size());

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

void HyperReduction::ComputeLinearPolynomialErrors(
    Hierarchy const& hierarchy,
    Eigen::Ref<MatrixX const> const& u)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.ComputeLinearPolynomialErrors");

    Data const& data = hierarchy.data;
    b.resizeLike(u);
    b.setZero();
    Ep.front().setZero();
    // Integrate element displacements
    using Quadrature  = math::SymmetricSimplexPolynomialQuadratureRule<kDims, 2 * kPolynomialOrder>;
    using Polynomial  = math::MonomialBasis<kDims, kPolynomialOrder>;
    using Tetrahedron = typename vbd::VolumeMesh::ElementType;
    auto nFineElements = eC.front().size();
    auto const& detJe  = data.wg;
    auto Ne            = fem::ShapeFunctions<Tetrahedron, 2 * kPolynomialOrder>();
    tbb::parallel_for(Index(0), nFineElements, [&](Index e) {
        auto const& ue = u(Eigen::placeholders::all, data.E.col(e));
        auto const& wg = common::ToEigen(Quadrature::weights);
        for (auto g = 0; g < Quadrature::kPoints; ++g)
        {
            auto ug = ue * Ne.col(g);
            b.col(e) += wg(g) * ug * detJe(e);
        }
    });
    // Solve cluster polynomial matching problems
    auto nLevels = Cptr.size();
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const& cptr     = Cptr[l];
        auto const& cadj     = Cadj[l];
        auto const nClusters = cptr.size() - 1;
        up[l].resize(Polynomial::kSize, nClusters);
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto cluster = cadj(Eigen::seq(cptr(c), cptr(c + 1) - 1));
            // Integrate cluster displacements
            auto bC  = b(Eigen::placeholders::all, cluster);
            b.col(c) = bC.colwise().sum();
            // Solve least-squares polynomial matching problem
            auto ApInvc  = ApInvC[l].block<4, 4>(0, 4 * c);
            up[l].col(c) = (ApInvc * b.col(c)).eval();
        });
    }
    // Integrate element polynomial errors
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>
#include <unordered_set>

TEST_CASE("[sim][vbd][multigrid] HyperReduction")
{
    using namespace pbat;
    using sim::vbd::Data;
    using sim::vbd::VolumeMesh;
    using sim::vbd::multigrid::Hierarchy;
    using sim::vbd::multigrid::HyperReduction;

    // Arrange
    auto const [VR, CR]   = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 2);
    Data data             = Data().WithVolumeMesh(VR, CR).Construct();
    auto const [VL2, CL2] = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 0);
    auto const [VL1, CL1] = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 1);
    Hierarchy H{std::move(data), {VolumeMesh(VL1, CL1), VolumeMesh(VL2, CL2)}};
    auto constexpr kClusterSize = 5;

    // Act
    HyperReduction HR{H, kClusterSize};

    // Assert
    auto nElements = H.data.E.cols();
    auto nLevels   = H.levels.size();
    SUBCASE("Each element/cluster is assigned to a single cluster at each parent level")
    {
        auto counts = IndexVectorX::Zero(nElements).eval();
        for (decltype(nLevels) l = 0; l < nLevels; ++l)
        {
            auto const& cptr = HR.Cptr[l];
            auto const& cadj = HR.Cadj[l];
            auto nClusters   = cptr.size() - 1;
            for (decltype(nClusters) c = 0; c < nClusters; ++c)
            {
                auto const& children = cadj(Eigen::seq(cptr(c), cptr(c + 1) - 1));
                counts(children).array() += 1;
            }
            auto const nChildren = cadj.size();
            bool const bAllChildrenHaveUniqueParentCluster =
                (counts(Eigen::seqN(0, nChildren)).array() == 1).all();
            CHECK(bAllChildrenHaveUniqueParentCluster);
            counts(Eigen::seqN(0, nChildren)).setZero();
            if (l == 0)
            {
                CHECK_EQ(nChildren, nElements);
            }
        }
    }
    SUBCASE("Cluster representatives have no duplicates at each level")
    {
        for (decltype(nLevels) l = 0; l < nLevels; ++l)
        {
            auto const nCluster         = HR.Cptr[l].size() - 1;
            auto const nRepresentatives = HR.eC[l].size();
            auto const nUnique = std::unordered_set<Index>(HR.eC[l].begin(), HR.eC[l].end()).size();
            CHECK_EQ(nCluster, nRepresentatives);
            CHECK_EQ(nRepresentatives, nUnique);
        }
    }
    SUBCASE("Per-cluster polynomial displacements have vanishing error") 
    {
        
    }
}
