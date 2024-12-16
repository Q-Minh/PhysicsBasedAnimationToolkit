#include "Integrator.h"

#include "Hierarchy.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Kernels.h"
#include "pbat/sim/vbd/lod/Smoother.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

void Integrator::Step(Scalar dt, Index substeps, Hierarchy& H) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Integrator.Step");

    using RootSmoother = pbat::sim::vbd::lod::Smoother;
    Scalar sdt         = dt / static_cast<Scalar>(substeps);
    Scalar sdt2        = sdt * sdt;
    auto nVertices     = H.data.x.cols();
    auto nElements     = H.data.mesh.E.cols();
    for (Index s = 0; s < substeps; ++s)
    {
        // Store previous positions
        H.data.xt = H.data.x;
        // Compute inertial target positions
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
        // Initialize block coordinate descent's, i.e. BCD's, solution
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
                // Compute element elasticities
                tbb::parallel_for(Index(0), nElements, [&](Index e) {
                    using pbat::math::linalg::mini::FromEigen;
                    using pbat::math::linalg::mini::ToEigen;
                    physics::StableNeoHookeanEnergy<3> Psi{};
                    Scalar mu        = H.data.lame(0, e);
                    Scalar lambda    = H.data.lame(1, e);
                    auto inds        = H.data.mesh.E(Eigen::placeholders::all, e);
                    Matrix<3, 4> xe  = H.data.x(Eigen::placeholders::all, inds);
                    Matrix<4, 3> GNe = H.data.GP.block<4, 3>(0, 3 * e);
                    Matrix<3, 3> F   = xe * GNe;
                    H.data.psiE(e)   = Psi.eval(FromEigen(F), mu, lambda);
                });

                // Update hyper reductions
                //for (auto& level : H.levels)
                //    level.HR.Update(H.data);
                auto lStl = static_cast<std::size_t>(l);
                H.levels[lStl].HR.Update(H.data);
                H.levels[lStl].Smooth(sdt, iters, H.data);
            }
        }
        // Update velocity
        H.data.vt = H.data.v;
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
