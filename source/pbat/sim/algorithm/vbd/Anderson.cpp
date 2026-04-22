#include "Anderson.h"

namespace pbat::sim::algorithm::vbd {

void AndersonParams::Serialize(io::Archive& archive, bool bMinimal) const
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.AndersonParams"];
    group.WriteMetaData("m", m);
    group.WriteMetaData("beta", beta);
    group.WriteMetaData("codNumericalZero", codNumericalZero);
    if (not bMinimal)
    {
        group.WriteData("Fk", Fk);
        group.WriteData("Xk", Xk);
        group.WriteData("xkm1", xkm1);
        group.WriteData("fk", fk);
        group.WriteData("fkm1", fkm1);
        group.WriteData("gammak", gammak);
    }
}

void AndersonParams::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.AndersonParams"];
    if (group.HasMetaData("m"))
        m = group.ReadMetaData<Index>("m");
    if (group.HasMetaData("beta"))
        beta = group.ReadMetaData<Scalar>("beta");
    if (group.HasMetaData("codNumericalZero"))
        codNumericalZero = group.ReadMetaData<Scalar>("codNumericalZero");
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

namespace {

struct AndersonTestSetup
{
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = pbat::sim::algorithm::common::FemElastoDynamics<ElasticEnergyType>;
    using ScalarType        = typename FemElastoDynamics::ScalarType;
    using IndexType         = typename FemElastoDynamics::IndexType;

    FemElastoDynamics dynamics;
    pbat::sim::algorithm::vbd::Params vbdParams;
    pbat::sim::algorithm::vbd::AndersonParams andersonParams;
    pbat::sim::contact::MeshDynamics<ScalarType, IndexType> meshDynamics;
    pbat::MatrixX X;
    pbat::geometry::Device device;
};

AndersonTestSetup SetupAndersonTest(pbat::Index maxIters = 10)
{
    using namespace pbat;
    AndersonTestSetup setup;

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

    // Anderson params
    setup.andersonParams.m                = 5;
    setup.andersonParams.beta             = Scalar(1);
    setup.andersonParams.codNumericalZero = Scalar(1e-10);

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

TEST_CASE("[sim][algorithm][vbd] Anderson")
{
    using namespace pbat;
    // Arrange
    auto setup = SetupAndersonTest(10);
    // Act
    setup.dynamics.SetInitialConditions(setup.dynamics.x, setup.dynamics.v);
    setup.dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = setup.dynamics.Objective(setup.dynamics.x);
    VectorX g0 = setup.dynamics.Gradient(setup.dynamics.x);
    sim::algorithm::vbd::InitializeSolve(
        setup.dynamics,
        setup.meshDynamics,
        setup.vbdParams,
        setup.andersonParams);
    sim::algorithm::vbd::Solve(
        setup.dynamics,
        setup.meshDynamics,
        setup.vbdParams,
        setup.andersonParams);
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

#include <tbb/global_control.h>

TEST_CASE("[sim][algorithm][anderson] Anderson Sandbox")
{
    // using namespace pbat;
    // auto archive            = io::Archive("sandbox.h5", HighFive::File::AccessMode::ReadOnly);
    // using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    // sim::algorithm::common::FemElastoDynamics<ElasticEnergyType> fem;
    // fem.Deserialize(archive["fem"]);
    // sim::contact::MeshDynamics<Scalar, Index> contact;
    // contact.Deserialize(archive["contact"]);
    // sim::algorithm::vbd::Params params;
    // params.Deserialize(archive["vbd/params"]);
    // sim::algorithm::vbd::AndersonParams andersonParams;
    // andersonParams.Deserialize(archive["vbd/anderson_params"]);
    // geometry::Device device{geometry::DeviceConfig{}.WithVerbosity(4)};
    // contact.Initialize(device);
    // tbb::global_control gc(tbb::global_control::max_allowed_parallelism, 1);
    // for (auto t = 0; t < 50; ++t)
    // {
    //     fem.SetupTimeIntegrationOptimization(
    //         sim::dynamics::EFemElastoDynamicsTimeStepInitialization::TrajectoryWithExternalLoad);
    //     sim::algorithm::vbd::InitializeSolve(fem, contact, params, andersonParams);
    //     sim::algorithm::vbd::Solve(fem, contact, params, andersonParams);
    //     fem.Step();
    // }
}