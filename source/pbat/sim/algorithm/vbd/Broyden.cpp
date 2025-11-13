#include "Broyden.h"

namespace pbat::sim::algorithm::vbd {

void BroydenParams::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.BroydenParams"];
    group.WriteMetaData("m", m);
    group.WriteMetaData("epsL2Solve", epsL2Solve);
    group.WriteMetaData("maxL2SolverIters", maxL2SolverIters);
    group.WriteMetaData("eL2Solver", static_cast<int>(eL2Solver));
    group.WriteMetaData("eJacobianEstimate", static_cast<int>(eJacobianEstimate));
    group.WriteMetaData("betaF", betaF);
    group.WriteMetaData("betaB", betaB);
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

void BroydenParams::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.BroydenParams"];
    m                 = group.ReadMetaData<Index>("m");
    epsL2Solve        = group.ReadMetaData<Scalar>("epsL2Solve");
    maxL2SolverIters  = group.ReadMetaData<Index>("maxL2SolverIters");
    eL2Solver = static_cast<EBroydenLeastSquaresSolver>(group.ReadMetaData<int>("eL2Solver"));
    eJacobianEstimate =
        static_cast<EBroydenJacobianEstimate>(group.ReadMetaData<int>("eJacobianEstimate"));
    betaF      = group.ReadMetaData<Scalar>("betaF");
    betaB      = group.ReadMetaData<Scalar>("betaB");
    k          = group.ReadMetaData<Index>("k");
    Fk         = group.ReadData<MatrixX>("Fk");
    Xk         = group.ReadData<MatrixX>("Xk");
    xkm1       = group.ReadData<VectorX>("xkm1");
    fk         = group.ReadData<VectorX>("fk");
    fkm1       = group.ReadData<VectorX>("fkm1");
    gammak     = group.ReadData<VectorX>("gammak");
    FkRowNorm2 = group.ReadData<MatrixX>("FkRowNorm2");
    Gkm        = group.ReadData<MatrixX>("Gkm");
    Sigma      = group.ReadData<VectorX>("Sigma");
    sqrtBetaB  = group.ReadMetaData<Scalar>("sqrtBetaB");
    Fknorm2    = group.ReadMetaData<Scalar>("Fknorm2");
    Bknorm2    = group.ReadMetaData<Scalar>("Bknorm2");
    gradL2     = group.ReadData<VectorX>("gradL2");
    FkgradL2   = group.ReadData<VectorX>("FkgradL2");
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

    FemElastoDynamics dynamics;
    pbat::sim::algorithm::vbd::Params vbdParams;
    std::shared_ptr<pbat::sim::algorithm::vbd::BroydenParams> broydenParams;
    pbat::sim::contact::MeshDynamics meshDynamics;
    pbat::MatrixX X;
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
    IndexMatrixX ilocal = IndexVector<4>{0, 1, 2, 3}.replicate(1, setup.dynamics.mesh.E.cols());
    auto GVT =
        graph::MeshAdjacencyMatrix(setup.dynamics.mesh.E, ilocal, setup.dynamics.mesh.X.cols());
    GVT                                = GVT.transpose();
    auto const [GVGp, GVGe, GVGilocal] = graph::MatrixToWeightedAdjacency(GVT);

    // Vertex colors
    auto GVV = graph::MeshPrimalGraph(setup.dynamics.mesh.E, setup.dynamics.mesh.X.cols());
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

    // Broyden params
    setup.broydenParams->m          = 5;
    setup.broydenParams->epsL2Solve = Scalar(1e-10);

    // Mesh contact dynamics (minimal setup for testing)
    IndexVectorX XCC(setup.X.cols()), ECC(C.cols()), Xord(setup.X.cols()), Eord(C.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(setup.X, C, XCC, ECC, Xord, Eord);
    graph::ReindexMeshByConnectedComponents(setup.X, C, XCC, ECC, Xord, Eord);
    sim::contact::MultiMesh<Index> multiMesh(C.bottomRows<4>(), XCC, nComponents);
    geometry::sdf::Forest<Scalar> sdfForest;
    sdfForest.nodes.push_back(
        geometry::sdf::Sphere<Scalar>{Scalar(10)}); // Large sphere to avoid contacts
    sdfForest.transforms.push_back(geometry::sdf::Transform<Scalar>::Identity());
    sdfForest.roots = {0};
    sdfForest.children.push_back({-1, -1});
    setup.meshDynamics.Construct(setup.X, std::move(multiMesh), std::move(sdfForest), Scalar(2));

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