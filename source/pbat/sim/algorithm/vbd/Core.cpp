#include "Core.h"

#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"

#include <exception>
#include <fmt/core.h>

namespace pbat::sim::algorithm::vbd {

void VertexElementAdjacencyGraph(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    Eigen::Ref<IndexVectorX> GVGp,
    Eigen::Ref<IndexVectorX> GVGe,
    Eigen::Ref<IndexVectorX> GVGilocal)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Core.VertexElementAdjacencyGraph");
    IndexMatrixX ilocal             = IndexVector<4>{0, 1, 2, 3}.replicate(1, E.cols());
    auto GVT                        = graph::MeshAdjacencyMatrix(E, ilocal, nNodes);
    GVT                             = GVT.transpose();
    std::tie(GVGp, GVGe, GVGilocal) = graph::MatrixToWeightedAdjacency(GVT);
}

void VertexColors(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    graph::EGreedyColorOrderingStrategy eOrdering,
    graph::EGreedyColorSelectionStrategy eSelection,
    Eigen::Ref<IndexVectorX> colors)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Core.VertexColors");
    auto GVV                = graph::MeshPrimalGraph(E, nNodes);
    auto [GVVp, GVVv, GVVw] = graph::MatrixToWeightedAdjacency(GVV);
    colors                  = graph::GreedyColor(GVVp, GVVv, eOrdering, eSelection);
}

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

Params&
Params::WithInitializationStrategy(dynamics::EFemElastoDynamicsTimeStepInitialization _strategy)
{
    eElasticsInitializationStrategy = _strategy;
    return *this;
}

PBAT_API Params& Params::WithDamping(Scalar _betaR)
{
    this->betaR = _betaR;
    return *this;
}

Params& Params::WithMaximumIterations(Index nIters)
{
    nMaxIters = nIters;
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

void Params::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.Params"];
    group.WriteData("GVGp", GVGp);
    group.WriteData("GVGe", GVGe);
    group.WriteData("GVGilocal", GVGilocal);
    group.WriteData("colors", colors);
    group.WriteData("Pptr", Pptr);
    group.WriteData("Padj", Padj);
    group.WriteMetaData(
        "eElasticsInitializationStrategy",
        static_cast<int>(eElasticsInitializationStrategy));
    group.WriteMetaData("detHZero", detHZero);
    group.WriteMetaData("nMaxIters", nMaxIters);
}

void Params::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.Params"];
    GVGp              = group.ReadData<IndexVectorX>("GVGp");
    GVGe              = group.ReadData<IndexVectorX>("GVGe");
    GVGilocal         = group.ReadData<IndexVectorX>("GVGilocal");
    colors            = group.ReadData<IndexVectorX>("colors");
    Pptr              = group.ReadData<IndexVectorX>("Pptr");
    Padj              = group.ReadData<IndexVectorX>("Padj");
    eElasticsInitializationStrategy =
        static_cast<dynamics::EFemElastoDynamicsTimeStepInitialization>(
            group.ReadMetaData<int>("eElasticsInitializationStrategy"));
    detHZero  = group.ReadMetaData<Scalar>("detHZero");
    nMaxIters = group.ReadMetaData<Index>("nMaxIters");
}

} // namespace pbat::sim::algorithm::vbd

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
    using FemElastoDynamics = pbat::sim::algorithm::common::FemElastoDynamics<ElasticEnergyType>;
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
    vbdParams.WithVertexElementAdjacencyGraph(GVGp, GVGe, GVGilocal)
        .WithVertexColors(colors)
        .WithMaximumIterations(10)
        .WithHessianDeterminantZeroUnder(Scalar{1e-6})
        .Construct();
    // Act
    dynamics.SetInitialConditions(dynamics.x, dynamics.v);
    Scalar f0  = dynamics.Objective(dynamics.x);
    VectorX g0 = dynamics.Gradient(dynamics.x);
    sim::algorithm::vbd::Solve(dynamics, vbdParams);
    // Assert
    auto constexpr zero = Scalar{1e-4};
    auto xt    = dynamics.bdf.CurrentState(0).reshaped(dynamics.x.rows(), dynamics.x.cols());
    MatrixX dx = dynamics.x - xt;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
    Scalar f = dynamics.Objective(dynamics.x);
    CHECK_LT(f, f0);
    VectorX g     = dynamics.Gradient(dynamics.x);
    Scalar g0norm = g0.norm();
    Scalar gnorm  = g.norm();
    CHECK_LT(gnorm, g0norm);
}