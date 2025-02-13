#include "HyperReduction.h"

#include "pbat/common/Eigen.h"
#include "pbat/common/Indexing.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/graph/Partition.h"
#include "pbat/math/PolynomialBasis.h"
#include "pbat/math/SymmetricQuadratureRules.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"
#include "pbat/sim/vbd/multigrid/Hierarchy.h"

#include <fmt/format.h>
#include <ranges>
#include <tbb/parallel_for.h>
#include <utility>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

using Polynomial = math::MonomialBasis<HyperReduction::kDims, HyperReduction::kPolynomialOrder>;
using Quadrature = math::SymmetricSimplexPolynomialQuadratureRule<
    HyperReduction::kDims,
    2 * HyperReduction::kPolynomialOrder>;
static auto constexpr kPolyCoeffs = static_cast<Index>(Polynomial::kSize);

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
    ComputeClusterQuadratureWeights(hierarchy);
    PrecomputeInversePolynomialMatrices(hierarchy);
}

void HyperReduction::AllocateWorkspace(std::size_t nLevels)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.AllocateWorkspace");
    ApInvC.resize(nLevels);
    bC.resize(nLevels + 1);
    wC.resize(nLevels);
    eC.resize(nLevels);
    up.resize(nLevels);
    Ep.resize(nLevels);
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const nClusters = Cptr[l].size() - 1;
        ApInvC[l].resize(kPolyCoeffs, kPolyCoeffs * nClusters);
        wC[l].resize(nClusters);
        eC[l].resize(nClusters);
        Ep[l].resize(nClusters);
    }
}

void HyperReduction::ConstructHierarchicalClustering(Hierarchy const& hierarchy, Index clusterSize)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.vbd.multigrid.HyperReduction.ConstructHierarchicalClustering");

    auto nLevels = hierarchy.levels.size();
    C.resize(nLevels);
    Cptr.resize(nLevels);
    Cadj.resize(nLevels);

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
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        // Reduce the graph via partitioning/clustering
        auto nGraphNodes = Gptr.size() - 1;
        auto nPartitions = nGraphNodes / clusterSize;
        C[l]             = graph::Partition(Gptr, Gadj, Gwts, nPartitions, opts);
        // Store the l^{th} level clustering
        std::tie(Cptr[l], Cadj[l]) = graph::MapToAdjacency(C[l]);
        // Exit if coarsest level reached
        if (l + 1 == nLevels)
            break;
        // Compute the supernodal graph (i.e. graph of clusters) as next graph to partition
        auto Gsizes = Gptr(Eigen::seqN(1, nGraphNodes)) - Gptr(Eigen::seqN(0, nGraphNodes));
        auto u = common::Repeat(IndexVectorX::LinSpaced(C[l].size(), 0, C[l].size() - 1), Gsizes);
        auto const& v = Gadj;
        auto SGu      = C[l](u);
        auto SGv      = C[l](v);
        auto edgeView = std::views::iota(Index(0), static_cast<Index>(SGu.size())) |
                        std::views::transform([&](Index i) {
                            return graph::WeightedEdge<Index, Index>(SGu(i), SGv(i), Gwts(i));
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

void HyperReduction::ComputeClusterQuadratureWeights(Hierarchy const& hierarchy)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.vbd.multigrid.HyperReduction.ComputeClusterQuadratureWeights");

    auto const& data = hierarchy.data;
    auto nLevels     = hierarchy.levels.size();
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const& cptr     = Cptr[l];
        auto const& cadj     = Cadj[l];
        auto const nClusters = cptr.size() - 1;
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto const& cluster = cadj(Eigen::seq(cptr(c), cptr(c + 1) - 1));
            auto const wg       = (l == 0) ? data.wg : wC[l - 1];
            wC[l](c)            = wg(cluster).sum();
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
    MatrixX Ap(kPolyCoeffs, kPolyCoeffs * nFineElements);
    Ap.setZero();

    // Compute the element polynomial inner product matrices
    tbb::parallel_for(Index(0), nFineElements, [&](Index e) {
        auto const& Xe = data.X(Eigen::placeholders::all, E.col(e));
        auto Xig       = common::ToEigen(Quadrature::points)
                       .reshaped(Quadrature::kDims + 1, Quadrature::kPoints);
        auto wg  = common::ToEigen(Quadrature::weights);
        auto Ape = Ap.block<kPolyCoeffs, kPolyCoeffs>(0, kPolyCoeffs * e);
        for (auto g = 0; g < Quadrature::kPoints; ++g)
        {
            auto Xg = Xe * Xig.col(g);
            auto Pg = Polynomial{}.eval(Xg);
            Ape += wg(g) * Pg * Pg.transpose() * detJe(e);
        }
    });

    // Compute the inverse of the per cluster polynomial inner product matrices
    auto nLevels = hierarchy.levels.size();
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const& cptr     = Cptr[l];
        auto const& cadj     = Cadj[l];
        auto const nClusters = cptr.size() - 1;
        ApInvC[l].setZero();
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto cluster = cadj(Eigen::seq(cptr(c), cptr(c + 1) - 1));
            auto ApInvc  = ApInvC[l].block<kPolyCoeffs, kPolyCoeffs>(0, kPolyCoeffs * c);
            for (auto cc : cluster)
            {
                auto const& App = (l == 0) ? Ap : ApInvC[l - 1];
                auto Apcc       = App.block<kPolyCoeffs, kPolyCoeffs>(0, kPolyCoeffs * cc);
                ApInvc += Apcc;
            }
        });
    }
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const nClusters = Cptr[l].size() - 1;
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto ApInvc = ApInvC[l].block<kPolyCoeffs, kPolyCoeffs>(0, kPolyCoeffs * c);
            ApInvc      = ApInvc.inverse().eval();
        });
    }
}

void HyperReduction::ComputeLinearPolynomialErrors(
    Hierarchy const& hierarchy,
    Eigen::Ref<MatrixX const> const& u)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.ComputeLinearPolynomialErrors");
    using Tetrahedron = typename vbd::VolumeMesh::ElementType;

    Data const& data = hierarchy.data;
    auto const dims  = u.rows();
    // Integrate element displacements
    auto nFineElements = C.front().size();
    bC.front().setZero(dims * kPolyCoeffs, nFineElements);
    auto const& detJe = data.wg;
    auto Ne           = fem::ShapeFunctions<Tetrahedron, 2 * kPolynomialOrder>();
    tbb::parallel_for(Index(0), nFineElements, [&](Index e) {
        auto const Xe  = data.X(Eigen::placeholders::all, data.E.col(e));
        auto const ue  = u(Eigen::placeholders::all, data.E.col(e));
        auto const wg  = common::ToEigen(Quadrature::weights);
        auto const Xig = common::ToEigen(Quadrature::points)
                             .reshaped(Quadrature::kDims + 1, Quadrature::kPoints);
        auto bCe = bC.front().col(e).reshaped(kPolyCoeffs, dims);
        for (auto g = 0; g < Quadrature::kPoints; ++g)
        {
            auto Xg = Xe * Xig.col(g);
            auto Pg = Polynomial{}.eval(Xg);
            auto ug = ue * Ne.col(g);
            // \int_{\Omega^e} P(X_g) u(X_g) \partial \Omega^e
            for (auto d = 0; d < dims; ++d)
                bCe.col(d) += wg(g) * Pg * ug(d) * detJe(e);
        }
    });
    // Solve cluster polynomial matching problems
    auto nLevels = Cptr.size();
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        auto const& cptr     = Cptr[l];
        auto const& cadj     = Cadj[l];
        auto const nClusters = cptr.size() - 1;
        up[l].resize(dims * kPolyCoeffs, nClusters);
        bC[l + 1].resize(dims * kPolyCoeffs, nClusters);
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto cluster = cadj(Eigen::seq(cptr(c), cptr(c + 1) - 1));
            // Integrate cluster displacements
            bC[l + 1].col(c) = bC[l](Eigen::placeholders::all, cluster).rowwise().sum();
            // Solve least-squares polynomial matching problem
            auto ApInvc = ApInvC[l].block<kPolyCoeffs, kPolyCoeffs>(0, kPolyCoeffs * c);
            auto Uplc   = up[l].col(c).reshaped(kPolyCoeffs, dims);
            auto Bc     = bC[l + 1].col(c).reshaped(kPolyCoeffs, dims);
            Uplc        = ApInvc * Bc;
        });
    }
    // Integrate cluster polynomial errors
    // \int_{\Omega^c} | P^c(X) u^c - u(X) |_2^2 \partial \Omega^c
    for (decltype(nLevels) l = 0; l < nLevels; ++l)
    {
        Ep[l].setZero();
        auto const& cptr     = Cptr[l];
        auto const& cadj     = Cadj[l];
        auto const nClusters = cptr.size() - 1;
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto cluster              = cadj(Eigen::seq(cptr(c), cptr(c + 1) - 1));
            auto Upc                  = up[l].col(c).reshaped(kPolyCoeffs, dims);
            bool const bIsFinestLevel = (l == 0);
            if (bIsFinestLevel)
            {
                for (Index e : cluster)
                {
                    auto const& Xe = data.X(Eigen::placeholders::all, data.E.col(e));
                    auto const& ue = u(Eigen::placeholders::all, data.E.col(e));
                    auto const& wg = common::ToEigen(Quadrature::weights);
                    for (auto g = 0; g < Quadrature::kPoints; ++g)
                    {
                        auto ug = ue * Ne.col(g);
                        auto Pg = Polynomial{}.eval(Xe * Ne.col(g));
                        Ep[l](c) += wg(g) * (Upc.transpose() * Pg - ug).squaredNorm() * detJe(e);
                    }
                }
                eC[l](c) = cluster(0);
            }
            else
            {
                for (Index cc : cluster)
                {
                    // Accumulate errors from child clusters
                    Ep[l](c) += Ep[l - 1](cc);
                    // Compute integrand at child cluster's representative element's centroid
                    Index e        = eC[l - 1](cc);
                    Scalar wgc     = wC[l - 1](cc);
                    auto const& Xe = data.X(Eigen::placeholders::all, data.E.col(e));
                    auto const& ue = u(Eigen::placeholders::all, data.E.col(e));
                    auto Xg        = Xe.rowwise().mean();
                    auto Pg        = Polynomial{}.eval(Xg);
                    auto ug        = ue.rowwise().mean();
                    // Accumulate this cluster's error w.r.t. its children
                    Ep[l](c) += wgc * (Upc.transpose() * Pg - ug).squaredNorm();
                }
                // Store the representative element of this cluster as this cluster's first child's
                // representative element
                eC[l](c) = eC[l - 1](cluster(0));
            }
        });
    }
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
    SUBCASE("Cluster quadrature weights reproduce the fine mesh's total weight")
    {
        Scalar const expectedTotalVolume = H.data.wg.sum();
        for (decltype(nLevels) l = 0; l < nLevels; ++l)
        {
            auto const& wc            = HR.wC[l];
            auto const volumeAtLevelL = wc.sum();
            CHECK_EQ(expectedTotalVolume, doctest::Approx(volumeAtLevelL));
        }
    }
    SUBCASE("Per-cluster polynomial displacements have vanishing error")
    {
        auto const fComputeAndCheckVanishingError = [&](Eigen::Ref<MatrixX const> const& u) {
            HR.ComputeLinearPolynomialErrors(H, u);
            for (decltype(nLevels) l = 0; l < nLevels; ++l)
            {
                auto const& Ep = HR.Ep[l];
                CHECK_EQ(Ep.maxCoeff(), doctest::Approx(0.0));
            }
        };
        auto nNodes = H.data.X.cols();
        SUBCASE("Displacement is translation")
        {
            MatrixX u = MatrixX::Ones(3, nNodes);
            fComputeAndCheckVanishingError(u);
        }
        SUBCASE("Displacement is scaling")
        {
            Vector<3> s = Vector<3>::Random();
            MatrixX u   = s.asDiagonal() * H.data.X - H.data.X;
            fComputeAndCheckVanishingError(u);
        }
    }
}
