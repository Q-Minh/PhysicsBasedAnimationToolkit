#include "Core.h"

#include "pbat/common/Atomic.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <fmt/core.h>
#include <thread>

namespace pbat::sim::algorithm::vbd {

void VertexElementAdjacencyGraph(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    IndexVectorX& GVGp,
    IndexVectorX& GVGe,
    IndexVectorX& GVGilocal)
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
    IndexVectorX& GVVp,
    IndexVectorX& GVVadj,
    IndexVectorX& colors)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Core.VertexColors");
    auto GVV               = graph::MeshPrimalGraph(E, nNodes);
    std::tie(GVVp, GVVadj) = graph::MatrixToAdjacency(GVV);
    colors                 = graph::GreedyColor(GVVp, GVVadj, eOrdering, eSelection);
}

void UpdatePenaltyParameter(contact::MeshDynamics<Scalar, Index>& contact, Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Core.UpdatePenaltyParameter");
    auto nThreads             = std::thread::hardware_concurrency();
    auto const& contactParams = contact.GetParams();
    auto nDynamicNodes        = params.xk.cols();
    std::atomic<Scalar> maxQ{0};
    contact.ForAllContacts(
        [&]<class TContactSet>(
            typename TContactSet::AccessorType C,
            auto&& stencil,
            std::int32_t /*t*/
        ) {
            auto nodes = contact.LoadStencil<TContactSet>(stencil);
            auto gradc = ToEigen(C.Grad());
            Scalar maxQc{0};
            for (auto ki = 0; ki < nodes.size(); ++ki)
            {
                auto i = nodes[ki];
                if (i >= nDynamicNodes)
                    continue;
                auto Hii    = params.Hk.template block<3, 3>(0, 3 * i);
                auto gradci = gradc.template segment<3>(ki * 3);
                Scalar Q    = gradci.dot(Hii * gradci) / gradci.squaredNorm();
                maxQc       = std::max(maxQc, Q);
            }
            switch (params.ePenaltyStiffness)
            {
                case ESALPenaltyStiffness::LocalMaxRayleighQuotient: {
                    auto F      = C.Friction();
                    C.Penalty() = contactParams.gamma * maxQc;
                    F.Penalty() = contactParams.gammaf * maxQc;
                }
                break;
                case ESALPenaltyStiffness::GlobalMaxRayleighQuotient: {
                    pbat::common::AtomicMax(maxQ, maxQc);
                }
                break;
                default: break;
            }
        },
        nThreads);
    if (params.ePenaltyStiffness == ESALPenaltyStiffness::GlobalMaxRayleighQuotient)
    {
        Scalar maxQv = maxQ.load(std::memory_order_relaxed);
        contact.ForAllContacts(
            [&]<class TContactSet>(
                typename TContactSet::AccessorType C,
                auto&& stencil,
                std::int32_t /*t*/
            ) {
                auto F      = C.Friction();
                C.Penalty() = contactParams.gamma * maxQv;
                F.Penalty() = contactParams.gammaf * maxQv;
            },
            nThreads);
    }
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

Params& Params::WithVertexColors(
    Eigen::Ref<IndexVectorX const> const& _GVVp,
    Eigen::Ref<IndexVectorX const> const& _GVVadj,
    Eigen::Ref<IndexVectorX const> const& _colors)
{
    GVVp                 = _GVVp;
    GVVadj               = _GVVadj;
    colors               = _colors;
    std::tie(Pptr, Padj) = graph::MapToAdjacency(colors);
    return *this;
}

Params& Params::WithDamping(Scalar _betaR)
{
    this->betaR = _betaR;
    return *this;
}

Params& Params::WithMaximumIterations(Index nIters)
{
    nMaxIters = nIters;
    return *this;
}

Params& Params::WithSubproblemMaximumIterations(Index nIters)
{
    nSubproblemMaxIters = nIters;
    return *this;
}

PBAT_API Params& Params::WithPenaltyParameterUpdateStrategy(ESALPenaltyStiffness strategy)
{
    ePenaltyStiffness = strategy;
    return *this;
}

Params& Params::WithStencilGradientAcceleration(
    Scalar _betaG0,
    Scalar _rhohat,
    Scalar _gammadown,
    Scalar _gammaup,
    bool _bWarmStartBeta,
    Scalar _wkinetic,
    Scalar _welastic,
    Scalar _wcontact)
{
    this->betaG0         = _betaG0;
    this->rhohat         = _rhohat;
    this->gammadown      = _gammadown;
    this->gammaup        = _gammaup;
    this->bWarmStartBeta = _bWarmStartBeta;
    this->wkinetic       = _wkinetic;
    this->welastic       = _welastic;
    this->wcontact       = _wcontact;
    return *this;
}

Params& Params::WithVertexLinearSolver(
    EVertexIntegrationLinearSolver solver,
    Scalar zero,
    Scalar eps,
    int iters)
{
    eSolver            = solver;
    hessZero           = zero;
    vLinSolverEps      = eps;
    vLinSolverMaxIters = iters;
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
        if (betaG0 < 0 or betaG0 >= 1)
        {
            throw std::invalid_argument(
                fmt::format(
                    "Initial stencil gradient augmentation coefficient betaG0 {} must satisfy 0 < "
                    "betaG0 < 1",
                    betaG0));
        }
        if (gammadown < 0 or gammadown >= 1)
        {
            throw std::invalid_argument(
                fmt::format(
                    "Stencil gradient beta reduction factor gammadown {} must satisfy 0 < "
                    "gammadown < 1",
                    gammadown));
        }
        if (gammaup < 0 or gammaup >= 1)
        {
            throw std::invalid_argument(
                fmt::format(
                    "Stencil gradient beta increase factor gammaup {} must satisfy 0 < gammaup < 1",
                    gammaup));
        }
        if (wkinetic < 0 or welastic < 0 or wcontact < 0)
        {
            throw std::invalid_argument(
                fmt::format(
                    "Weight factors must be non-negative: wkinetic {}, welastic {}, wcontact "
                    "{}",
                    wkinetic,
                    welastic,
                    wcontact));
        }
    }
    xb.resize(3, nVerts);
    gkinetic.resize(3, nVerts);
    gelastic.resize(3, nVerts);
    gcontact.resize(3, nVerts);
    xk.resize(3, nVerts);
    Hnk.resize(nVerts);
    betaG.resize(nVerts);
    Hk.resize(3, 3 * nVerts);
    return *this;
}

void Params::Serialize(io::Archive& archive, bool bMinimal) const
{
    io::Archive group = archive["pbat.sim.algorithm.vbd.Params"];
    group.WriteMetaData("betaR", betaR);
    group.WriteMetaData("nMaxIters", nMaxIters);
    group.WriteMetaData("nSubproblemMaxIters", nSubproblemMaxIters);
    group.WriteMetaData("gtol", gtol);
    group.WriteMetaData("ePenaltyStiffness", static_cast<int>(ePenaltyStiffness));
    group.WriteMetaData("eSolver", static_cast<int>(eSolver));
    group.WriteMetaData("hessZero", hessZero);
    group.WriteMetaData("vLinSolverEps", vLinSolverEps);
    group.WriteMetaData("vLinSolverMaxIters", vLinSolverMaxIters);
    group.WriteMetaData("betaG0", betaG0);
    group.WriteMetaData("rhohat", rhohat);
    group.WriteMetaData("gammadown", gammadown);
    group.WriteMetaData("gammaup", gammaup);
    group.WriteMetaData("bWarmStartBeta", static_cast<int>(bWarmStartBeta));
    group.WriteMetaData("wkinetic", wkinetic);
    group.WriteMetaData("welastic", welastic);
    group.WriteMetaData("wcontact", wcontact);
    if (not bMinimal)
    {
        group.WriteData("GVGp", GVGp);
        group.WriteData("GVGe", GVGe);
        group.WriteData("GVGilocal", GVGilocal);
        group.WriteData("colors", colors);
        group.WriteData("GVVp", GVVp);
        group.WriteData("GVVadj", GVVadj);
        group.WriteData("Pptr", Pptr);
        group.WriteData("Padj", Padj);
        group.WriteData("xb", xb);
        group.WriteData("gkinetic", gkinetic);
        group.WriteData("gelastic", gelastic);
        group.WriteData("gcontact", gcontact);
        group.WriteData("xk", xk);
        group.WriteData("Hnk", Hnk);
        group.WriteData("betaG", betaG);
        group.WriteData("Hk", Hk);
        group.WriteMetaData("k", k);
        group.WriteMetaData("kp", kp);
    }
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
    if (group.HasData("GVVp"))
        GVVp = group.ReadData<decltype(GVVp)>("GVVp");
    if (group.HasData("GVVadj"))
        GVVadj = group.ReadData<decltype(GVVadj)>("GVVadj");
    if (group.HasData("Pptr"))
        Pptr = group.ReadData<decltype(Pptr)>("Pptr");
    if (group.HasData("Padj"))
        Padj = group.ReadData<decltype(Padj)>("Padj");
    if (group.HasMetaData("betaR"))
        betaR = group.ReadMetaData<decltype(betaR)>("betaR");
    if (group.HasMetaData("nMaxIters"))
        nMaxIters = group.ReadMetaData<decltype(nMaxIters)>("nMaxIters");
    if (group.HasMetaData("nSubproblemMaxIters"))
        nSubproblemMaxIters =
            group.ReadMetaData<decltype(nSubproblemMaxIters)>("nSubproblemMaxIters");
    if (group.HasMetaData("gtol"))
        gtol = group.ReadMetaData<decltype(gtol)>("gtol");
    if (group.HasMetaData("ePenaltyStiffness"))
        ePenaltyStiffness =
            static_cast<decltype(ePenaltyStiffness)>(group.ReadMetaData<int>("ePenaltyStiffness"));
    if (group.HasMetaData("eSolver"))
        eSolver = static_cast<decltype(eSolver)>(group.ReadMetaData<int>("eSolver"));
    if (group.HasMetaData("hessZero"))
        hessZero = group.ReadMetaData<decltype(hessZero)>("hessZero");
    if (group.HasMetaData("vLinSolverEps"))
        vLinSolverEps = group.ReadMetaData<decltype(vLinSolverEps)>("vLinSolverEps");
    if (group.HasMetaData("vLinSolverMaxIters"))
        vLinSolverMaxIters = group.ReadMetaData<decltype(vLinSolverMaxIters)>("vLinSolverMaxIters");
    if (group.HasMetaData("betaG0"))
        betaG0 = group.ReadMetaData<decltype(betaG0)>("betaG0");
    if (group.HasMetaData("rhohat"))
        rhohat = group.ReadMetaData<decltype(rhohat)>("rhohat");
    if (group.HasMetaData("gammadown"))
        gammadown = group.ReadMetaData<decltype(gammadown)>("gammadown");
    if (group.HasMetaData("gammaup"))
        gammaup = group.ReadMetaData<decltype(gammaup)>("gammaup");
    if (group.HasMetaData("bWarmStartBeta"))
        bWarmStartBeta = static_cast<bool>(group.ReadMetaData<int>("bWarmStartBeta"));
    if (group.HasMetaData("wkinetic"))
        wkinetic = group.ReadMetaData<decltype(wkinetic)>("wkinetic");
    if (group.HasMetaData("welastic"))
        welastic = group.ReadMetaData<decltype(welastic)>("welastic");
    if (group.HasMetaData("wcontact"))
        wcontact = group.ReadMetaData<decltype(wcontact)>("wcontact");
    if (group.HasData("xb"))
        xb = group.ReadData<decltype(xb)>("xb");
    if (group.HasData("gkinetic"))
        gkinetic = group.ReadData<decltype(gkinetic)>("gkinetic");
    if (group.HasData("gelastic"))
        gelastic = group.ReadData<decltype(gelastic)>("gelastic");
    if (group.HasData("gcontact"))
        gcontact = group.ReadData<decltype(gcontact)>("gcontact");
    if (group.HasData("xk"))
        xk = group.ReadData<decltype(xk)>("xk");
    if (group.HasData("Hnk"))
        Hnk = group.ReadData<decltype(Hnk)>("Hnk");
    if (group.HasData("betaG"))
        betaG = group.ReadData<decltype(betaG)>("betaG");
    if (group.HasData("Hk"))
        Hk = group.ReadData<decltype(Hk)>("Hk");
    if (group.HasMetaData("k"))
        k = group.ReadMetaData<decltype(k)>("k");
    if (group.HasMetaData("kp"))
        kp = group.ReadMetaData<decltype(kp)>("kp");
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
        .WithSubproblemMaximumIterations(maxIters)
        .WithVertexLinearSolver(sim::algorithm::vbd::EVertexIntegrationLinearSolver::Inverse)
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

TEST_CASE("[type:integration][sim][algorithm][vbd] Cube sliding on plane")
{
    using namespace pbat;
    using namespace pbat::sim::algorithm;
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using FemElastoDynamics = sim::algorithm::common::FemElastoDynamics<ElasticEnergyType>;
    using MeshDynamics      = sim::contact::MeshDynamics<Scalar, Index>;
    // Arrange
    io::Archive archive(
        fmt::format("{}/sim/algorithm/CubeFallingOnPlaneFast.h5", PBAT_TESTS_INTEGRATION_PATH),
        HighFive::File::AccessMode::ReadOnly);
    FemElastoDynamics fem{};
    fem.Deserialize(archive["fem"]);
    MeshDynamics contact{};
    contact.Deserialize(archive["contact"]);
    geometry::Device device{geometry::DeviceConfig{}};
    contact.Initialize(device);
    contact.GetParams()
        .WithOgcParams(
            sim::contact::ogc::Params<Scalar>()
                .WithDisplacementBoundConfig(0.45, 0.)
                .WithRadii(1e-2 /*r*/, 1e-2 /*rq*/)
                .Construct())
        .Construct();
    sim::algorithm::vbd::Params params;
    sim::algorithm::vbd::VertexElementAdjacencyGraph(
        fem.mesh.E,
        fem.mesh.X.cols(),
        params.GVGp,
        params.GVGe,
        params.GVGilocal);
    // Vertex colors
    auto eOrdering  = graph::EGreedyColorOrderingStrategy::LargestDegree;
    auto eSelection = graph::EGreedyColorSelectionStrategy::LeastUsed;
    sim::algorithm::vbd::VertexColors(
        fem.mesh.E,
        fem.mesh.X.cols(),
        eOrdering,
        eSelection,
        params.GVVp,
        params.GVVadj,
        params.colors);
    // VBD params
    params.WithVertexColors(params.GVVp, params.GVVadj, params.colors).Construct();
    // Act
    for (auto t = 0; t < 200; ++t)
    {
        fem.SetupTimeIntegrationOptimization();
        sim::algorithm::vbd::InitializeSolve(fem, contact, params);
        bool const bHasContacts = contact.NumContacts() > 0;
        sim::algorithm::vbd::Solve(fem, contact, params);
        fem.Step();
    }
}

TEST_CASE("[type:debug][sim][algorithm][vbd] Sandbox")
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
    auto eOrdering  = graph::EGreedyColorOrderingStrategy::LargestDegree;
    auto eSelection = graph::EGreedyColorSelectionStrategy::LeastUsed;
    sim::algorithm::vbd::VertexColors(
        fem.mesh.E,
        fem.mesh.X.cols(),
        eOrdering,
        eSelection,
        params.GVVp,
        params.GVVadj,
        params.colors);
    // VBD params
    params.WithVertexColors(params.GVVp, params.GVVadj, params.colors).Construct();
    geometry::Device device{geometry::DeviceConfig{}};
    contact.Initialize(device);
    contact.GetParams().Construct();
    auto initStrategy = static_cast<sim::dynamics::EFemElastoDynamicsTimeStepInitialization>(
        archive.ReadMetaData<int>("initialization_strategy"));
    for (auto t = 0; t < 2; ++t)
    {
        fem.SetupTimeIntegrationOptimization(initStrategy);
        sim::algorithm::vbd::InitializeSolve(fem, contact, params);
        sim::algorithm::vbd::Solve(fem, contact, params);
        fem.Step();
    }
}