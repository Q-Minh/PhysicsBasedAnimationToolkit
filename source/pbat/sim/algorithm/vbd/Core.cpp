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

PBAT_API Params& Params::WithHomogenization(EHomogenizationStrategy strategy, Scalar _betac)
{
    this->eHomogenizationStrategy = strategy;
    this->betac                   = _betac;
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
        if (betac <= 0)
        {
            throw std::invalid_argument(
                fmt::format(
                    "Contact homogenization conditioning factor betac {} must be positive",
                    betac));
        }
    }
    xb.resize(3, nVerts);
    gamma.resize(5, nVerts);
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
    group.WriteMetaData("detHZero", detHZero);
    group.WriteMetaData("nMaxIters", nMaxIters);
    group.WriteMetaData("eHomogenizationStrategy", static_cast<int>(eHomogenizationStrategy));
    group.WriteMetaData("betac", betac);
    group.WriteData("xb", xb);
    group.WriteData("gamma", gamma);
    group.WriteMetaData("k", k);
}

void Params::Deserialize(io::Archive const& archive)
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.Params"];
    if (group.HasData("GVGp"))
        GVGp = group.ReadData<decltype(GVGp)>("GVGp");
    if (group.HasData("GVGe"))
        GVGe = group.ReadData<decltype(GVGe)>("GVGe");
    if (group.HasData("GVGilocal"))
        GVGilocal = group.ReadData<decltype(GVGilocal)>("GVGilocal");
    if (group.HasData("colors"))
        colors = group.ReadData<decltype(colors)>("colors");
    if (group.HasData("Pptr"))
        Pptr = group.ReadData<decltype(Pptr)>("Pptr");
    if (group.HasData("Padj"))
        Padj = group.ReadData<decltype(Padj)>("Padj");
    if (group.HasData("detHZero"))
        detHZero = group.ReadMetaData<decltype(detHZero)>("detHZero");
    if (group.HasMetaData("nMaxIters"))
        nMaxIters = group.ReadMetaData<decltype(nMaxIters)>("nMaxIters");
    if (group.HasMetaData("eHomogenizationStrategy"))
        eHomogenizationStrategy = static_cast<EHomogenizationStrategy>(
            group.ReadMetaData<int>("eHomogenizationStrategy"));
    if (group.HasMetaData("betac"))
        betac = group.ReadMetaData<decltype(betac)>("betac");
    if (group.HasData("xb"))
        xb = group.ReadData<decltype(xb)>("xb");
    if (group.HasData("gamma"))
        gamma = group.ReadData<decltype(gamma)>("gamma");
    if (group.HasMetaData("k"))
        k = group.ReadMetaData<decltype(k)>("k");
}

} // namespace pbat::sim::algorithm::vbd

#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <doctest/doctest.h>

namespace {

struct VbdTestSetup
{
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = pbat::sim::algorithm::common::FemElastoDynamics<ElasticEnergyType>;
    using ScalarType        = typename FemElastoDynamics::ScalarType;
    using IndexType         = typename FemElastoDynamics::IndexType;

    FemElastoDynamics dynamics;
    pbat::sim::algorithm::vbd::Params vbdParams;
    pbat::sim::contact::MeshDynamics<ScalarType, IndexType> meshDynamics;
    pbat::MatrixX X;
    pbat::geometry::Device device;
};

VbdTestSetup SetupVbdTest(pbat::Index maxIters = 10)
{
    using namespace pbat;
    VbdTestSetup setup;

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

TEST_CASE("[sim][algorithm][vbd] Core")
{
    using namespace pbat;
    // Arrange
    auto setup = SetupVbdTest(10);
    // Act
    setup.dynamics.SetInitialConditions(setup.dynamics.x, setup.dynamics.v);
    setup.dynamics.SetupTimeIntegrationOptimization();
    Scalar f0  = setup.dynamics.Objective(setup.dynamics.x);
    VectorX g0 = setup.dynamics.Gradient(setup.dynamics.x);
    sim::algorithm::vbd::InitializeSolve(setup.dynamics, setup.meshDynamics, setup.vbdParams);
    sim::algorithm::vbd::Solve(setup.dynamics, setup.meshDynamics, setup.vbdParams);
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

TEST_CASE("[sim][algorithm][vbd] Sandbox")
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
    // geometry::Device device{geometry::DeviceConfig{}.WithVerbosity(4)};
    // contact.Initialize(device);
    // contact.GetParams()
    //     .WithNormalContact(contact.GetParams().kc)
    //     .WithFrictionalContact(contact.GetParams().mu, contact.GetParams().epsv);
    // contact.ComputeDisplacementBounds(fem.x);
    // for (auto t = 0; t < 50; ++t)
    // {
    //     fem.SetupTimeIntegrationOptimization(
    //         sim::dynamics::EFemElastoDynamicsTimeStepInitialization::TrajectoryWithExternalLoad);
    //     contact.TruncateDisplacedPositions(fem.x, fem.dmask);
    //     sim::algorithm::vbd::Solve(fem, contact, params);
    //     fem.Step();
    // }
}