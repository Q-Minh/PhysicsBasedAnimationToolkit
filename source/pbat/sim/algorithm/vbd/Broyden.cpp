#include "Broyden.h"

namespace pbat::sim::algorithm::vbd {

void BroydenParams::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.BroydenParams"];
    group.WriteData("m", m);
    group.WriteData("epsL2Solve", epsL2Solve);
    group.WriteData("maxL2SolverIters", maxL2SolverIters);
    group.WriteData("eL2Solver", static_cast<int>(eL2Solver));
    group.WriteData("eJacobianEstimate", static_cast<int>(eJacobianEstimate));
    group.WriteData("betaF", betaF);
    group.WriteData("betaB", betaB);
    group.WriteData("k", k);
    group.WriteData("Fk", Fk);
    group.WriteData("Xk", Xk);
    group.WriteData("xkm1", xkm1);
    group.WriteData("fk", fk);
    group.WriteData("fkm1", fkm1);
    group.WriteData("gammak", gammak);
    group.WriteData("FkRowNorm2", FkRowNorm2);
    group.WriteData("Gkm", Gkm);
    group.WriteData("Sigma", Sigma);
    group.WriteData("sqrtBetaB", sqrtBetaB);
    group.WriteData("Fknorm2", Fknorm2);
    group.WriteData("Bknorm2", Bknorm2);
    group.WriteData("gradL2", gradL2);
    group.WriteData("FkgradL2", FkgradL2);
}

void BroydenParams::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.BroydenParams"];
    m                 = group.ReadData<Index>("m");
    epsL2Solve        = group.ReadData<Scalar>("epsL2Solve");
    maxL2SolverIters  = group.ReadData<Index>("maxL2SolverIters");
    eL2Solver         = static_cast<EBroydenLeastSquaresSolver>(group.ReadData<int>("eL2Solver"));
    eJacobianEstimate =
        static_cast<EBroydenJacobianEstimate>(group.ReadData<int>("eJacobianEstimate"));
    betaF      = group.ReadData<Scalar>("betaF");
    betaB      = group.ReadData<Scalar>("betaB");
    k          = group.ReadData<Index>("k");
    Fk         = group.ReadData<MatrixX>("Fk");
    Xk         = group.ReadData<MatrixX>("Xk");
    xkm1       = group.ReadData<VectorX>("xkm1");
    fk         = group.ReadData<VectorX>("fk");
    fkm1       = group.ReadData<VectorX>("fkm1");
    gammak     = group.ReadData<VectorX>("gammak");
    FkRowNorm2 = group.ReadData<MatrixX>("FkRowNorm2");
    Gkm        = group.ReadData<MatrixX>("Gkm");
    Sigma      = group.ReadData<VectorX>("Sigma");
    sqrtBetaB  = group.ReadData<Scalar>("sqrtBetaB");
    Fknorm2    = group.ReadData<Scalar>("Fknorm2");
    Bknorm2    = group.ReadData<Scalar>("Bknorm2");
    gradL2     = group.ReadData<VectorX>("gradL2");
    FkgradL2   = group.ReadData<VectorX>("FkgradL2");
}

} // namespace pbat::sim::algorithm::vbd

#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][algorithm][vbd] Broyden")
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
    // Broyden params
    sim::algorithm::vbd::BroydenParams broydenParams{};
    broydenParams.m          = 5;
    broydenParams.epsL2Solve = Scalar(1e-10);
    // Act
    dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = dynamics.Objective();
    VectorX g0 = dynamics.Gradient();
    sim::algorithm::vbd::Solve(dynamics, vbdParams, broydenParams);
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