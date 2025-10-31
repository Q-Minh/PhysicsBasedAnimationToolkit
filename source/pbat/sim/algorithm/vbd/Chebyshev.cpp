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

TEST_CASE("[sim][algorithm][vbd] Chebyshev")
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
    vbdParams.WithVertexElementAdjacencyGraph(GVGp, GVGe, GVGilocal)
        .WithVertexColors(colors)
        .WithMaximumIterations(20)
        .WithHessianDeterminantZeroUnder(Scalar{1e-6})
        .Construct();
    // Chebyshev params
    sim::algorithm::vbd::ChebyshevParams chebyshevParams{};
    chebyshevParams.rho = Scalar(0.9);
    // Act
    dynamics.SetInitialConditions(dynamics.x, dynamics.v);
    dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = dynamics.Objective(dynamics.x);
    VectorX g0 = dynamics.Gradient(dynamics.x);
    sim::algorithm::vbd::Solve(dynamics, vbdParams, chebyshevParams);
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