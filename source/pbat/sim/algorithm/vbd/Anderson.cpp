#include "Anderson.h"

namespace pbat::sim::algorithm::vbd {

void AndersonParams::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.AndersonParams"];
    group.WriteMetaData("m", m);
    group.WriteMetaData("beta", beta);
    group.WriteMetaData("codNumericalZero", codNumericalZero);
    group.WriteMetaData("k", k);
    group.WriteData("Fk", Fk);
    group.WriteData("Xk", Xk);
    group.WriteData("xkm1", xkm1);
    group.WriteData("fk", fk);
    group.WriteData("fkm1", fkm1);
    group.WriteData("gammak", gammak);
}

void AndersonParams::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.AndersonParams"];
    m                 = group.ReadMetaData<Index>("m");
    beta              = group.ReadMetaData<Scalar>("beta");
    codNumericalZero  = group.ReadMetaData<Scalar>("codNumericalZero");
    k                 = group.ReadMetaData<Index>("k");
    Fk                = group.ReadData<MatrixX>("Fk");
    Xk                = group.ReadData<MatrixX>("Xk");
    xkm1              = group.ReadData<VectorX>("xkm1");
    fk                = group.ReadData<VectorX>("fk");
    fkm1              = group.ReadData<VectorX>("fkm1");
    gammak            = group.ReadData<VectorX>("gammak");
}

void AndersonParams::AllocateIfNeeded(Index n)
{
    Fk.resize(n, m);
    Xk.resize(n, m);
    xkm1.resize(n);
    fk.resize(n);
    fkm1.resize(n);
    gammak.resize(m);
}

} // namespace pbat::sim::algorithm::vbd

#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][algorithm][vbd] Anderson")
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
        .WithMaximumIterations(10)
        .WithHessianDeterminantZeroUnder(Scalar{1e-6})
        .Construct();
    // Anderson params
    sim::algorithm::vbd::AndersonParams andersonParams{};
    andersonParams.m                = 5;
    andersonParams.beta             = Scalar(1);
    andersonParams.codNumericalZero = Scalar(1e-10);
    // Act
    dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = dynamics.Objective();
    VectorX g0 = dynamics.Gradient();
    sim::algorithm::vbd::Solve(dynamics, vbdParams, andersonParams);
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