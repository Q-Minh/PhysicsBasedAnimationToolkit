#include "Broyden.h"

namespace pbat::sim::algorithm::vbd {

void BroydenParams::Serialize(io::Archive& archive, bool bMinimal) const
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.BroydenParams"];
    group.WriteMetaData("m", m);
    group.WriteMetaData("epsL2Solve", epsL2Solve);
    group.WriteMetaData("maxL2SolverIters", maxL2SolverIters);
    group.WriteMetaData("eL2Solver", static_cast<int>(eL2Solver));
    group.WriteMetaData("eJacobianEstimate", static_cast<int>(eJacobianEstimate));
    group.WriteMetaData("betaF", betaF);
    group.WriteMetaData("betaB", betaB);
    group.WriteMetaData("nMaxIters", nMaxIters);
    if (not bMinimal)
    {
        group.WriteMetaData("k", k);
        group.WriteData("Fk", Fk);
        group.WriteData("Xk", Xk);
        group.WriteData("xkm1", xkm1);
        group.WriteData("fk", fk);
        group.WriteData("fkm1", fkm1);
        group.WriteData("gammak", gammak);
        group.WriteData("FkRowNorm2", FkRowNorm2);
        group.WriteData("Gkm", Gkm);
        group.WriteData("Sigma", Sigma);
        group.WriteMetaData("sqrtBetaB", sqrtBetaB);
        group.WriteMetaData("Fknorm2", Fknorm2);
        group.WriteMetaData("Bknorm2", Bknorm2);
        group.WriteData("gradL2", gradL2);
        group.WriteData("FkgradL2", FkgradL2);
    }
}

void BroydenParams::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.BroydenParams"];
    if (group.HasMetaData("m"))
        m = group.ReadMetaData<Index>("m");
    if (group.HasMetaData("epsL2Solve"))
        epsL2Solve = group.ReadMetaData<Scalar>("epsL2Solve");
    if (group.HasMetaData("maxL2SolverIters"))
        maxL2SolverIters = group.ReadMetaData<Index>("maxL2SolverIters");
    if (group.HasMetaData("eL2Solver"))
        eL2Solver =
            static_cast<EBroydenLeastSquaresSolver>(group.ReadMetaData<int>("eL2Solver"));
    if (group.HasMetaData("eJacobianEstimate"))
        eJacobianEstimate =
            static_cast<EBroydenJacobianEstimate>(group.ReadMetaData<int>("eJacobianEstimate"));
    if (group.HasMetaData("betaF"))
        betaF = group.ReadMetaData<Scalar>("betaF");
    if (group.HasMetaData("betaB"))
        betaB = group.ReadMetaData<Scalar>("betaB");
    if (group.HasMetaData("nMaxIters"))
        nMaxIters = group.ReadMetaData<Index>("nMaxIters");
    if (group.HasMetaData("k"))
        k = group.ReadMetaData<Index>("k");
    if (group.HasData("Fk"))
        Fk = group.ReadData<MatrixX>("Fk");
    if (group.HasData("Xk"))
        Xk = group.ReadData<MatrixX>("Xk");
    if (group.HasData("xkm1"))
        xkm1 = group.ReadData<VectorX>("xkm1");
    if (group.HasData("fk"))
        fk = group.ReadData<VectorX>("fk");
    if (group.HasData("fkm1"))
        fkm1 = group.ReadData<VectorX>("fkm1");
    if (group.HasData("gammak"))
        gammak = group.ReadData<VectorX>("gammak");
    if (group.HasData("FkRowNorm2"))
        FkRowNorm2 = group.ReadData<MatrixX>("FkRowNorm2");
    if (group.HasData("Gkm"))
        Gkm = group.ReadData<MatrixX>("Gkm");
    if (group.HasData("Sigma"))
        Sigma = group.ReadData<VectorX>("Sigma");
    if (group.HasMetaData("sqrtBetaB"))
        sqrtBetaB = group.ReadMetaData<Scalar>("sqrtBetaB");
    if (group.HasMetaData("Fknorm2"))
        Fknorm2 = group.ReadMetaData<Scalar>("Fknorm2");
    if (group.HasMetaData("Bknorm2"))
        Bknorm2 = group.ReadMetaData<Scalar>("Bknorm2");
    if (group.HasData("gradL2"))
        gradL2 = group.ReadData<VectorX>("gradL2");
    if (group.HasData("FkgradL2"))
        FkgradL2 = group.ReadData<VectorX>("FkgradL2");
}

} // namespace pbat::sim::algorithm::vbd

#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>
#include <memory>

namespace {

struct BroydenTestSetup
{
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = pbat::sim::algorithm::common::FemElastoDynamics<ElasticEnergyType>;
    using ScalarType        = typename FemElastoDynamics::ScalarType;
    using IndexType         = typename FemElastoDynamics::IndexType;

    FemElastoDynamics dynamics;
    pbat::sim::algorithm::vbd::Params vbdParams;
    std::shared_ptr<pbat::sim::algorithm::vbd::BroydenParams> broydenParams;
    pbat::sim::contact::MeshDynamics<ScalarType, IndexType> meshDynamics;
    pbat::MatrixX X;
    pbat::geometry::Device device;
};

BroydenTestSetup SetupBroydenTest(pbat::Index maxIters = 10)
{
    using namespace pbat;
    BroydenTestSetup setup;

    // Cube mesh
    setup.X.resize(3, 8);
    IndexMatrixX C(4, 5);
    // clang-format off
    setup.X << 0., 1., 0., 1., 0., 1., 0., 1.,
               0., 0., 1., 1., 0., 0., 1., 1.,
               0., 0., 0., 0., 1., 1., 1., 1.;
    C << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on

    // Construct FEM dynamics
    setup.dynamics.Construct(setup.X, C);

    // Adjacency structures
    sim::algorithm::vbd::VertexElementAdjacencyGraph(
        C,
        setup.X.cols(),
        setup.vbdParams.GVGp,
        setup.vbdParams.GVGe,
        setup.vbdParams.GVGilocal);

    // Vertex colors
    auto eOrdering  = graph::EGreedyColorOrderingStrategy::LargestDegree;
    auto eSelection = graph::EGreedyColorSelectionStrategy::LeastUsed;
    sim::algorithm::vbd::VertexColors(
        C,
        setup.X.cols(),
        graph::EGreedyColorOrderingStrategy::LargestDegree,
        graph::EGreedyColorSelectionStrategy::LeastUsed,
        setup.vbdParams.GVVp,
        setup.vbdParams.GVVadj,
        setup.vbdParams.colors);

    // VBD params
    setup.vbdParams
        .WithVertexColors(setup.vbdParams.GVVp, setup.vbdParams.GVVadj, setup.vbdParams.colors)
        .WithMaximumIterations(maxIters)
        .WithHessianSingularUnder(Scalar{1e-6})
        .Construct();

    // Broyden params
    setup.broydenParams             = std::make_shared<sim::algorithm::vbd::BroydenParams>();
    setup.broydenParams->m          = 5;
    setup.broydenParams->epsL2Solve = Scalar(1e-10);

    // Mesh contact dynamics (minimal setup for testing)
    IndexVectorX XCC(setup.X.cols()), ECC(C.cols()), Xord(setup.X.cols()), Eord(C.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(setup.X, C, XCC, ECC, Xord, Eord);
    graph::ReindexMeshByConnectedComponents(setup.X, C, XCC, ECC, Xord, Eord);
    sim::contact::MultiMesh<Index> multiMesh(C.bottomRows<4>(), XCC, nComponents);
    setup.meshDynamics.SetDynamicGeometry(setup.X, std::move(multiMesh));
    setup.meshDynamics.Initialize(setup.device);
    return setup;
}

} // namespace

TEST_CASE("[sim][algorithm][vbd] Broyden")
{
    using namespace pbat;
    // Arrange
    auto setup = SetupBroydenTest(10);
    // Act
    setup.dynamics.SetInitialConditions(setup.dynamics.x, setup.dynamics.v);
    setup.dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = setup.dynamics.Objective(setup.dynamics.x);
    VectorX g0 = setup.dynamics.Gradient(setup.dynamics.x);
    sim::algorithm::vbd::InitializeSolve(
        setup.dynamics,
        setup.meshDynamics,
        setup.vbdParams,
        *setup.broydenParams);
    sim::algorithm::vbd::Solve(
        setup.dynamics,
        setup.meshDynamics,
        setup.vbdParams,
        *setup.broydenParams);
    // Assert
    auto constexpr zero = Scalar{1e-4};
    auto xt             = setup.dynamics.bdf.CurrentState(0).reshaped(
        setup.dynamics.x.rows(),
        setup.dynamics.x.cols());
    MatrixX dx                           = setup.dynamics.x - xt;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
    Scalar f = setup.dynamics.Objective(setup.dynamics.x);
    CHECK_LT(f, f0);
    VectorX g     = setup.dynamics.Gradient(setup.dynamics.x);
    Scalar g0norm = g0.norm();
    Scalar gnorm  = g.norm();
    CHECK_LT(gnorm, g0norm);
}

TEST_CASE("[sim][algorithm][broyden] Sandbox")
{
    using namespace pbat;
    auto archive            = io::Archive("sandbox.h5", HighFive::File::AccessMode::ReadOnly);
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    sim::algorithm::common::FemElastoDynamics<ElasticEnergyType> fem;
    fem.Deserialize(archive["fem"]);
    sim::contact::MeshDynamics<Scalar, Index> contact;
    contact.Deserialize(archive["contact"]);
    sim::algorithm::vbd::Params params;
    params.Deserialize(archive["vbd/params"]);
    sim::algorithm::vbd::BroydenParams broyden;
    broyden.Deserialize(archive["vbd/broyden_params"]);
    geometry::Device device{geometry::DeviceConfig{}.WithVerbosity(1)};
    contact.Initialize(device);
    contact.GetParams().Construct();
    // fem.SetupTimeIntegrationOptimization(
    //     sim::dynamics::EFemElastoDynamicsTimeStepInitialization::TrajectoryWithExternalLoad);
    sim::algorithm::vbd::InitializeSolve(fem, contact, params, broyden);
    sim::algorithm::vbd::Solve(fem, contact, params, broyden);
}