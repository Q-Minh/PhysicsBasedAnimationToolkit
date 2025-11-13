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

namespace {

struct AndersonTestSetup
{
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = pbat::sim::algorithm::common::FemElastoDynamics<ElasticEnergyType>;

    FemElastoDynamics dynamics;
    pbat::sim::algorithm::vbd::Params vbdParams;
    pbat::sim::algorithm::vbd::AndersonParams andersonParams;
    pbat::sim::contact::MeshDynamics meshDynamics;
    pbat::MatrixX X;
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
    IndexMatrixX ilocal = IndexVector<4>{0, 1, 2, 3}.replicate(1, setup.dynamics.mesh.E.cols());
    auto GVT = graph::MeshAdjacencyMatrix(setup.dynamics.mesh.E, ilocal, setup.dynamics.mesh.X.cols());
    GVT      = GVT.transpose();
    auto const [GVGp, GVGe, GVGilocal] = graph::MatrixToWeightedAdjacency(GVT);

    // Vertex colors
    auto GVV                = graph::MeshPrimalGraph(setup.dynamics.mesh.E, setup.dynamics.mesh.X.cols());
    auto [GVVp, GVVv, GVVw] = graph::MatrixToWeightedAdjacency(GVV);
    auto eOrdering          = graph::EGreedyColorOrderingStrategy::LargestDegree;
    auto eSelection         = graph::EGreedyColorSelectionStrategy::LeastUsed;
    auto colors             = graph::GreedyColor(GVVp, GVVv, eOrdering, eSelection);

    // VBD params
    setup.vbdParams.WithVertexElementAdjacencyGraph(GVGp, GVGe, GVGilocal)
        .WithVertexColors(colors)
        .WithMaximumIterations(maxIters)
        .WithHessianDeterminantZeroUnder(Scalar{1e-6})
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
    geometry::sdf::Forest<Scalar> sdfForest;
    sdfForest.nodes.push_back(geometry::sdf::Sphere<Scalar>{Scalar(10)}); // Large sphere to avoid contacts
    sdfForest.transforms.push_back(geometry::sdf::Transform<Scalar>::Identity());
    sdfForest.roots = {0};
    sdfForest.children.push_back({-1, -1});
    setup.meshDynamics.Construct(setup.X, std::move(multiMesh), std::move(sdfForest), Scalar(2));

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
    sim::algorithm::vbd::Solve(setup.dynamics, setup.meshDynamics, setup.vbdParams, setup.andersonParams);
    // Assert
    auto constexpr zero = Scalar{1e-4};
    auto xt    = setup.dynamics.bdf.CurrentState(0).reshaped(setup.dynamics.x.rows(), setup.dynamics.x.cols());
    MatrixX dx = setup.dynamics.x - xt;
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