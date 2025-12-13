#include "Chebyshev.h"

namespace pbat::sim::algorithm::vbd {

void ChebyshevParams::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.ChebyshevParams"];
    group.WriteMetaData("rho", rho);
    group.WriteMetaData("k", k);
    group.WriteMetaData("rho2", rho2);
    group.WriteMetaData("omega", omega);
    group.WriteData("xkm1", xkm1);
    group.WriteData("xkm2", xkm2);
}

void ChebyshevParams::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.ChebyshevParams"];
    rho               = group.ReadMetaData<Scalar>("rho");
    k                 = group.ReadMetaData<Index>("k");
    rho2              = group.ReadMetaData<Scalar>("rho2");
    omega             = group.ReadMetaData<Scalar>("omega");
    xkm1              = group.ReadData<Eigen::Matrix<Scalar, 3, Eigen::Dynamic>>("xkm1");
    xkm2              = group.ReadData<Eigen::Matrix<Scalar, 3, Eigen::Dynamic>>("xkm2");
}

} // namespace pbat::sim::algorithm::vbd

#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>

namespace {

struct ChebyshevTestSetup
{
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = pbat::sim::algorithm::common::FemElastoDynamics<ElasticEnergyType>;
    using ScalarType        = typename FemElastoDynamics::ScalarType;
    using IndexType         = typename FemElastoDynamics::IndexType;

    FemElastoDynamics dynamics;
    pbat::sim::algorithm::vbd::Params vbdParams;
    pbat::sim::algorithm::vbd::ChebyshevParams chebyshevParams;
    pbat::sim::contact::MeshDynamics<ScalarType, IndexType> meshDynamics;
    pbat::MatrixX X;
    pbat::geometry::Device device;
    pbat::sim::contact::MeshDynamicsParams<ScalarType, IndexType> meshDynamicsParams;
};

ChebyshevTestSetup SetupChebyshevTest(pbat::Index maxIters = 20)
{
    using namespace pbat;
    ChebyshevTestSetup setup;

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

    // Chebyshev params
    setup.chebyshevParams.rho = Scalar(0.9);

    // Mesh contact dynamics (minimal setup for testing)
    IndexVectorX XCC(setup.X.cols()), ECC(C.cols()), Xord(setup.X.cols()), Eord(C.cols());
    Eigen::Index const nComponents =
        graph::SortedConnectedComponentOrdering(setup.X, C, XCC, ECC, Xord, Eord);
    graph::ReindexMeshByConnectedComponents(setup.X, C, XCC, ECC, Xord, Eord);
    sim::contact::MultiMesh<Index> multiMesh(C.bottomRows<4>(), XCC, nComponents);
    setup.meshDynamics.SetDynamicGeometry(setup.X, std::move(multiMesh));
    setup.meshDynamics.Initialize(setup.device, setup.meshDynamicsParams);
    return setup;
}

} // namespace

TEST_CASE("[sim][algorithm][vbd] Chebyshev")
{
    using namespace pbat;
    // Arrange
    auto setup = SetupChebyshevTest(20);
    // Act
    setup.dynamics.SetInitialConditions(setup.dynamics.x, setup.dynamics.v);
    setup.dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = setup.dynamics.Objective(setup.dynamics.x);
    VectorX g0 = setup.dynamics.Gradient(setup.dynamics.x);
    sim::algorithm::vbd::Solve(
        setup.dynamics,
        setup.meshDynamics,
        setup.vbdParams,
        setup.chebyshevParams);
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