#include "Core.h"

#include "pbat/graph/Adjacency.h"

#include <exception>
#include <fmt/core.h>

namespace pbat::sim::algorithm::vbd {

Params& Params::WithVertexElementAdjacencyGraph(
    Eigen::Ref<IndexVectorX const> const& _GVGp,
    Eigen::Ref<IndexVectorX const> const& _GVGe,
    Eigen::Ref<IndexVectorX const> const& _GVGilocal)
{
    GVGp      = _GVGp;
    GVGe      = _GVGe;
    GVGilocal = _GVGilocal;
    return *this;
}

Params& Params::WithVertexColors(Eigen::Ref<IndexVectorX const> const& _colors)
{
    colors               = _colors;
    std::tie(Pptr, Padj) = graph::MapToAdjacency(colors);
    return *this;
}

Params& Params::WithInitializationStrategy(EInitializationStrategy _strategy)
{
    strategy = _strategy;
    return *this;
}

Params& Params::WithHessianDeterminantZeroUnder(Scalar zero)
{
    detHZero = zero;
    return *this;
}

Params& Params::Construct(bool bValidate)
{
    if (bValidate)
    {
        auto nVerts = colors.size();
        if (GVGp.size() != nVerts + 1)
        {
            throw std::invalid_argument(
                fmt::format(
                    "GVGp size {} inconsistent with expected # verts {}",
                    GVGp.size(),
                    nVerts));
        }
        if (Padj.size() != nVerts)
        {
            throw std::invalid_argument(
                fmt::format(
                    "Padj size {} inconsistent with expected # verts {}",
                    Padj.size(),
                    nVerts));
        }
        auto nPartitions = Pptr.size() - 1;
        if (colors.maxCoeff() + 1 != nPartitions)
        {
            throw std::invalid_argument(
                fmt::format(
                    "# colors {} inconsistent with expected # partitions {}",
                    colors.maxCoeff() + 1,
                    nPartitions));
        }
        auto nVertexElementAdjacencies = GVGe.size();
        if (GVGilocal.size() != nVertexElementAdjacencies)
        {
            throw std::invalid_argument(
                fmt::format(
                    "GVGilocal size {} inconsistent with expected # vertex-element adjacencies {}",
                    GVGilocal.size(),
                    nVertexElementAdjacencies));
        }
    }
    return *this;
}

} // namespace pbat::sim::algorithm::vbd

#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][algorithm][vbd] Core")
{
    using namespace pbat;
    // Arrange
    // Cube mesh
    MatrixX V(3, 8);
    IndexMatrixX C(4, 5);
    // clang-format off
    V << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    C << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    // Problem parameters
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = pbat::sim::algorithm::vbd::FemElastoDynamics<ElasticEnergyType>;
    FemElastoDynamics dynamics{};
    dynamics.Construct(V, C);
    // Adjacency structures
    sim::algorithm::vbd::Params vbdParams{};
    IndexMatrixX ilocal = IndexVector<4>{0, 1, 2, 3}.replicate(1, dynamics.mesh.E.cols());
    auto GVT = graph::MeshAdjacencyMatrix(dynamics.mesh.E, ilocal, dynamics.mesh.X.cols());
    GVT      = GVT.transpose();
    auto const [GVGp, GVGe, GVGilocal] = graph::MatrixToWeightedAdjacency(GVT);
    // Vertex colors
    auto GVV                = graph::MeshPrimalGraph(dynamics.mesh.E, dynamics.mesh.X.cols());
    auto [GVVp, GVVv, GVVw] = graph::MatrixToWeightedAdjacency(GVV);
    auto eOrdering          = graph::EGreedyColorOrderingStrategy::LargestDegree;
    auto eSelection         = graph::EGreedyColorSelectionStrategy::LeastUsed;
    auto colors             = graph::GreedyColor(GVVp, GVVv, eOrdering, eSelection);
    // Initialization strategy
    auto eInitializationStrategy = pbat::sim::algorithm::vbd::EInitializationStrategy::Inertia;
    vbdParams.WithInitializationStrategy(eInitializationStrategy)
        .WithVertexElementAdjacencyGraph(GVGp, GVGe, GVGilocal)
        .WithVertexColors(colors)
        .WithHessianDeterminantZeroUnder(Scalar{1e-6})
        .Construct();
    // Act
    auto constexpr iterations = 10;
    dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = dynamics.Objective();
    VectorX g0 = dynamics.Gradient();
    sim::algorithm::vbd::InitializeSolve(dynamics, vbdParams);
    for (auto k = 0; k < iterations; ++k)
        sim::algorithm::vbd::Step(dynamics, vbdParams);
    // Assert
    auto constexpr zero = Scalar{1e-4};
    auto xt    = dynamics.bdf.CurrentState(0).reshaped(dynamics.x.rows(), dynamics.x.cols());
    MatrixX dx = dynamics.x - xt;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
    dynamics.ComputeElasticEnergy(
        fem::EElementElasticityComputationFlags::Potential |
            fem::EElementElasticityComputationFlags::Gradient,
        fem::EHyperElasticSpdCorrection::None);
    Scalar f = dynamics.Objective();
    CHECK_LT(f, f0);
    VectorX g     = dynamics.Gradient();
    Scalar g0norm = g0.norm();
    Scalar gnorm  = g.norm();
    CHECK_LT(gnorm, g0norm);
}