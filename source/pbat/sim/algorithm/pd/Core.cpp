#include "Core.h"

#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"

#include <exception>
#include <fmt/core.h>

namespace pbat::sim::algorithm::pd {

void CellElementAdjacencyGraph(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    Eigen::Ref<IndexVectorX> GTGp,
    Eigen::Ref<IndexVectorX> GTGe,
    Eigen::Ref<IndexVectorX> GTGilocal)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.pd.Core.VertexElementAdjacencyGraph");
    IndexMatrixX ilocal             = IndexVector<4>{0, 1, 2, 3}.replicate(1, E.cols());
    auto GVT                        = graph::MeshAdjacencyMatrix(E, ilocal, nNodes);
    GVT                             = GVT.transpose();
    std::tie(GTGp, GTGe, GTGilocal) = graph::MatrixToWeightedAdjacency(GVT);
}

void CellColors(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    graph::EGreedyColorOrderingStrategy eOrdering,
    graph::EGreedyColorSelectionStrategy eSelection,
    Eigen::Ref<IndexVectorX> colors)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.pd.Core.VertexColors");
    auto GVV                = graph::MeshPrimalGraph(E, nNodes);
    auto [GVVp, GVVv, GVVw] = graph::MatrixToWeightedAdjacency(GVV);
    colors                  = graph::GreedyColor(GVVp, GVVv, eOrdering, eSelection);
}

Params& Params::WithCellElementAdjacencyGraph(
    Eigen::Ref<IndexVectorX const> const& _GVGp,
    Eigen::Ref<IndexVectorX const> const& _GVGe,
    Eigen::Ref<IndexVectorX const> const& _GVGilocal)
{
    GTGp      = _GVGp;
    GTGe      = _GVGe;
    GTGilocal = _GVGilocal;
    return *this;
}

Params& Params::WithCellColors(Eigen::Ref<IndexVectorX const> const& _colors)
{
    colors               = _colors;
    std::tie(Pptr, Padj) = graph::MapToAdjacency(colors);
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
    auto nVerts = colors.size();
    if (bValidate)
    {
        if (GTGp.size() != nVerts + 1)
        {
            throw std::invalid_argument(
                fmt::format(
                    "GTGp size {} inconsistent with expected # verts {}",
                    GTGp.size(),
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
        auto nVertexElementAdjacencies = GTGe.size();
        if (GTGilocal.size() != nVertexElementAdjacencies)
        {
            throw std::invalid_argument(
                fmt::format(
                    "GTGilocal size {} inconsistent with expected # vertex-element adjacencies {}",
                    GTGilocal.size(),
                    nVertexElementAdjacencies));
        }
    }
    xb.resize(3, nVerts);
    return *this;
}

void Params::Serialize(io::Archive& archive, bool bMinimal) const
{
    io::Archive group = archive["pbat.sim.algorithm.pd.Params"];
    group.WriteMetaData("detHZero", detHZero);
    group.WriteMetaData("nMaxIters", nMaxIters);
    if (not bMinimal)
    {
        group.WriteData("GTGp", GTGp);
        group.WriteData("GTGe", GTGe);
        group.WriteData("GTGilocal", GTGilocal);
        group.WriteData("colors", colors);
        group.WriteData("Pptr", Pptr);
        group.WriteData("Padj", Padj);
        group.WriteData("xb", xb);
    }
}

void Params::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.pd.Params"];
    if (group.HasData("GTGp"))
        GTGp = group.ReadData<IndexVectorX>("GTGp");
    if (group.HasData("GTGe"))
        GTGe = group.ReadData<IndexVectorX>("GTGe");
    if (group.HasData("GTGilocal"))
        GTGilocal = group.ReadData<IndexVectorX>("GTGilocal");
    if (group.HasData("colors"))
        colors = group.ReadData<IndexVectorX>("colors");
    if (group.HasData("Pptr"))
        Pptr = group.ReadData<IndexVectorX>("Pptr");
    if (group.HasData("Padj"))
        Padj = group.ReadData<IndexVectorX>("Padj");
    if (group.HasMetaData("detHZero"))
        detHZero = group.ReadMetaData<Scalar>("detHZero");
    if (group.HasMetaData("nMaxIters"))
        nMaxIters = group.ReadMetaData<Index>("nMaxIters");
    if (group.HasData("xb"))
        xb = group.ReadData<MatrixX>("xb");
}

} // namespace pbat::sim::algorithm::pd

#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>

namespace {

struct PdTestSetup
{
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = pbat::sim::algorithm::common::FemElastoDynamics<ElasticEnergyType>;
    using ScalarType        = typename FemElastoDynamics::ScalarType;
    using IndexType         = typename FemElastoDynamics::IndexType;

    FemElastoDynamics dynamics;
    pbat::sim::algorithm::pd::Params pdParams;
    pbat::sim::contact::MeshDynamics<ScalarType, IndexType> meshDynamics;
    pbat::MatrixX X;
    pbat::geometry::Device device;
};

PdTestSetup SetupPdTest(pbat::Index maxIters = 10)
{
    // using namespace pbat;
    // PdTestSetup setup;

    // // Cube mesh
    // setup.X.resize(3, 8);
    // IndexMatrixX C(4, 5);
    // // clang-format off
    // setup.X << 0., 1., 0., 1., 0., 1., 0., 1.,
    //            0., 0., 1., 1., 0., 0., 1., 1.,
    //            0., 0., 0., 0., 1., 1., 1., 1.;
    // C << 0, 3, 5, 6, 0,
    //      1, 2, 4, 7, 5,
    //      3, 0, 6, 5, 3,
    //      5, 6, 0, 3, 6;
    // // clang-format on

    // // Construct FEM dynamics
    // setup.dynamics.Construct(setup.X, C);

    // // Adjacency structures
    // IndexMatrixX ilocal = IndexVector<4>{0, 1, 2, 3}.replicate(1, setup.dynamics.mesh.E.cols());
    // auto GVT =
    //     graph::MeshAdjacencyMatrix(setup.dynamics.mesh.E, ilocal, setup.dynamics.mesh.X.cols());
    // GVT                                = GVT.transpose();
    // auto const [GVGp, GVGe, GVGilocal] = graph::MatrixToWeightedAdjacency(GVT);

    // // Vertex colors
    // auto GVV = graph::MeshPrimalGraph(setup.dynamics.mesh.E, setup.dynamics.mesh.X.cols());
    // auto [GVVp, GVVv, GVVw] = graph::MatrixToWeightedAdjacency(GVV);
    // auto eOrdering          = graph::EGreedyColorOrderingStrategy::LargestDegree;
    // auto eSelection         = graph::EGreedyColorSelectionStrategy::LeastUsed;
    // auto colors             = graph::GreedyColor(GVVp, GVVv, eOrdering, eSelection);

    // // PD params
    // setup.pdParams.WithCellElementAdjacencyGraph(GVGp, GVGe, GVGilocal)
    //     .WithCellColors(colors)
    //     .WithMaximumIterations(maxIters)
    //     .WithHessianDeterminantZeroUnder(Scalar{1e-6})
    //     .Construct();

    // // Mesh contact dynamics (minimal setup for testing)
    // IndexVectorX XCC(setup.X.cols()), ECC(C.cols()), Xord(setup.X.cols()), Eord(C.cols());
    // Eigen::Index const nComponents =
    //     graph::SortedConnectedComponentOrdering(setup.X, C, XCC, ECC, Xord, Eord);
    // graph::ReindexMeshByConnectedComponents(setup.X, C, XCC, ECC, Xord, Eord);
    // sim::contact::MultiMesh<Index> multiMesh(C.bottomRows<4>(), XCC, nComponents);
    // setup.meshDynamics.SetDynamicGeometry(setup.X, std::move(multiMesh));
    // setup.meshDynamics.Initialize(setup.device);
    // return setup;
}

} // namespace

TEST_CASE("[sim][algorithm][pd] Core")
{
    // using namespace pbat;
    // // Arrange
    // auto setup = SetupPdTest(10);
    // // Act
    // setup.dynamics.SetInitialConditions(setup.dynamics.x, setup.dynamics.v);
    // setup.dynamics.SetupTimeIntegrationOptimization();
    // Scalar f0  = setup.dynamics.Objective(setup.dynamics.x);
    // VectorX g0 = setup.dynamics.Gradient(setup.dynamics.x);
    // sim::algorithm::pd::Solve(setup.dynamics, setup.meshDynamics, setup.pdParams);
    // // Assert
    // auto constexpr zero = Scalar{1e-4};
    // auto xt             = setup.dynamics.bdf.CurrentState(0).reshaped(
    //     setup.dynamics.x.rows(),
    //     setup.dynamics.x.cols());
    // MatrixX dx                           = setup.dynamics.x - xt;
    // bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    // CHECK(bVerticesFallUnderGravity);
    // bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    // CHECK(bVerticesOnlyFall);
    // Scalar f = setup.dynamics.Objective(setup.dynamics.x);
    // CHECK_LT(f, f0);
    // VectorX g     = setup.dynamics.Gradient(setup.dynamics.x);
    // Scalar g0norm = g0.norm();
    // Scalar gnorm  = g.norm();
    // CHECK_LT(gnorm, g0norm);
}

TEST_CASE("[sim][algorithm][pd] Sandbox")
{
    // using namespace pbat;
    // auto archive            = io::Archive("sandbox.h5", HighFive::File::AccessMode::ReadOnly);
    // using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    // sim::algorithm::common::FemElastoDynamics<ElasticEnergyType> fem;
    // fem.Deserialize(archive["fem"]);
    // sim::contact::MeshDynamics<Scalar, Index> contact;
    // contact.Deserialize(archive["contact"]);
    // sim::algorithm::pd::Params params;
    // params.Deserialize(archive["pd/params"]);
    // geometry::Device device{geometry::DeviceConfig{}.WithVerbosity(4)};
    // contact.Initialize(device);
    // Scalar const zPlane = contact.StaticPointPositions().row(2).minCoeff();
    // contact.GetParams()
    //     .WithNormalContact(contact.GetParams().kc)
    //     .WithFrictionalContact(contact.GetParams().mu, contact.GetParams().epsv);
    // contact.ComputeDisplacementBounds(fem.x);
    // for (auto t = 0; t < 50; ++t)
    // {
    //     fem.SetupTimeIntegrationOptimization(
    //         sim::dynamics::EFemElastoDynamicsTimeStepInitialization::TrajectoryWithExternalLoad);
    //     contact.TruncateDisplacedPositions(fem.x, fem.dmask);
    //     sim::algorithm::pd::Solve(fem, contact, params);
    //     fem.Step();
    //     Scalar const zMin = fem.x.row(2).minCoeff();
    //     CHECK_GT(zMin, zPlane);
    // }
}