#include "Integrator.h"

#include "Hierarchy.h"
#include "pbat/common/ArgSort.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Kernels.h"
#include "pbat/sim/vbd/multigrid/Smoother.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

void Integrator::ComputeAndSortStrainRates(
    [[maybe_unused]] Hierarchy& H,
    [[maybe_unused]] Scalar sdt) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Integrator.ComputeAndSortStrainRates");
    // auto nElements = H.data.E.cols();
    // tbb::parallel_for(Index(0), nElements, [&](Index e) {
    //     auto inds      = H.data.E(Eigen::placeholders::all, e);
    //     auto xe        = H.data.x(Eigen::placeholders::all, inds);
    //     auto GNe       = H.data.GP.block<4, 3>(0, 3 * e);
    //     Matrix<3, 3> F = xe * GNe;
    //     // E = 1/2 (F^T F - I)
    //     Matrix<3, 3> E = F.transpose() * F;
    //     E.diagonal().array() -= Scalar(1);
    //     E *= Scalar(0.5);
    //     auto GreenStrain       = H.data.mGreenStrains.block<3, 3>(0, 3 * e);
    //     auto GreenStrainAtT    = H.data.mGreenStrainsAtT.block<3, 3>(0, 3 * e);
    //     GreenStrainAtT         = GreenStrain;
    //     GreenStrain            = E;
    //     Matrix<3, 3> Edot      = (GreenStrain - GreenStrainAtT) / sdt;
    //     H.data.mStrainRates(e) = Edot.norm();
    // });
    // std::sort(
    //     H.data.mStrainRateOrder.begin(),
    //     H.data.mStrainRateOrder.end(),
    //     [&](Index ei, Index ej) { return H.data.mStrainRates(ei) < H.data.mStrainRates(ej); });
}

void Integrator::ComputeInertialTargetPositions(Hierarchy& H, Scalar sdt, Scalar sdt2) const
{
    auto nVertices = H.data.x.cols();
    tbb::parallel_for(Index(0), nVertices, [&](Index i) {
        using pbat::sim::vbd::kernels::InertialTarget;
        using pbat::math::linalg::mini::FromEigen;
        using pbat::math::linalg::mini::ToEigen;
        auto xtilde = InertialTarget(
            FromEigen(H.data.xt.col(i).head<3>()),
            FromEigen(H.data.v.col(i).head<3>()),
            FromEigen(H.data.aext.col(i).head<3>()),
            sdt,
            sdt2);
        H.data.xtilde.col(i) = ToEigen(xtilde);
    });
}

void Integrator::InitializeBCD(Hierarchy& H, Scalar sdt, Scalar sdt2) const
{
    auto nVertices = H.data.x.cols();
    tbb::parallel_for(Index(0), nVertices, [&](Index i) {
        using pbat::sim::vbd::kernels::InitialPositionsForSolve;
        using pbat::math::linalg::mini::FromEigen;
        using pbat::math::linalg::mini::ToEigen;
        auto x = InitialPositionsForSolve(
            FromEigen(H.data.xt.col(i).head<3>()),
            FromEigen(H.data.vt.col(i).head<3>()),
            FromEigen(H.data.v.col(i).head<3>()),
            FromEigen(H.data.aext.col(i).head<3>()),
            sdt,
            sdt2,
            H.data.strategy);
        H.data.x.col(i) = ToEigen(x);
    });
}

void Integrator::UpdateVelocity(Hierarchy& H, Scalar sdt) const
{
    auto nVertices = H.data.x.cols();
    H.data.vt      = H.data.v;
    tbb::parallel_for(Index(0), nVertices, [&](Index i) {
        using pbat::sim::vbd::kernels::IntegrateVelocity;
        using pbat::math::linalg::mini::FromEigen;
        using pbat::math::linalg::mini::ToEigen;
        auto v = IntegrateVelocity(
            FromEigen(H.data.xt.col(i).head<3>()),
            FromEigen(H.data.x.col(i).head<3>()),
            sdt);
        H.data.v.col(i) = ToEigen(v);
    });
}

void Integrator::Step(Scalar dt, Index substeps, Hierarchy& H) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Integrator.Step");

    using RootSmoother = pbat::sim::vbd::multigrid::Smoother;
    Scalar sdt         = dt / static_cast<Scalar>(substeps);
    Scalar sdt2        = sdt * sdt;
    for (Index s = 0; s < substeps; ++s)
    {
        ComputeAndSortStrainRates(H, sdt);
        // Store previous positions
        H.data.xt = H.data.x;
        ComputeInertialTargetPositions(H, sdt, sdt2);
        InitializeBCD(H, sdt, sdt2);
        // Hierarchical solve
        auto nLevelVisits = H.cycle.size();
        for (auto c = 0; c < nLevelVisits; ++c)
        {
            Index l     = H.cycle(c);
            Index iters = H.siters(c);
            if (l < 0)
            {
                RootSmoother{}.Apply(iters, sdt, H.data);
            }
            else
            {
                auto lStl = static_cast<std::size_t>(l);
                H.levels[lStl].Smooth(sdt, iters, H.data);
            }
        }
        UpdateVelocity(H, sdt);
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][multigrid] Integrator")
{
    using namespace pbat;
    using sim::vbd::Data;
    using sim::vbd::VolumeMesh;
    using sim::vbd::multigrid::Hierarchy;
    using sim::vbd::multigrid::Integrator;

    // Arrange
    Scalar dt{1e-2};
    Index substeps{1};
    auto const [VR, CR]   = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 2);
    auto const [VL1, CL1] = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 1);
    auto const [VL2, CL2] = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 0);
    IndexVectorX cycle(3);
    cycle << 1, 0, -1;
    IndexVectorX siters(3);
    siters << 2, 1, 1;
    Hierarchy H{
        Data()
            .WithVolumeMesh(VR, CR)
            .WithInitializationStrategy(sim::vbd::EInitializationStrategy::KineticEnergyMinimum)
            .Construct(),
        {VolumeMesh(VL1, CL1), VolumeMesh(VL2, CL2)},
        cycle,
        siters};

    // Act
    Integrator mvbd{};
    mvbd.Step(dt, substeps, H);

    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = H.data.x - VR;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}
