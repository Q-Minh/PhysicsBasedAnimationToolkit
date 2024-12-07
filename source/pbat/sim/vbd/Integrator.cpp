#include "Integrator.h"

#include "Kernels.h"
#include "Smoother.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/profiling/Profiling.h"

#include <tbb/parallel_for.h>
#include <type_traits>

namespace pbat {
namespace sim {
namespace vbd {

Integrator::Integrator(Data dataIn) : data(std::move(dataIn)) {}

void Integrator::Step(Scalar dt, Index iterations, Index substeps, Scalar rho)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.Integrator.Step");

    Scalar sdt           = dt / (static_cast<Scalar>(substeps));
    Scalar sdt2          = sdt * sdt;
    auto const nVertices = data.x.cols();
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::ToEigen;
    for (auto s = 0; s < substeps; ++s)
    {
        // Store previous positions
        data.xt = data.x;
        // Compute inertial target positions
        tbb::parallel_for(Index(0), nVertices, [&](Index i) {
            auto xtilde = kernels::InertialTarget(
                FromEigen(data.xt.col(i).head<3>()),
                FromEigen(data.v.col(i).head<3>()),
                FromEigen(data.aext.col(i).head<3>()),
                sdt,
                sdt2);
            data.xtilde.col(i) = ToEigen(xtilde);
        });
        // Initialize block coordinate descent's, i.e. BCD's, solution
        tbb::parallel_for(Index(0), nVertices, [&](Index i) {
            auto x = kernels::InitialPositionsForSolve(
                FromEigen(data.xt.col(i).head<3>()),
                FromEigen(data.vt.col(i).head<3>()),
                FromEigen(data.v.col(i).head<3>()),
                FromEigen(data.aext.col(i).head<3>()),
                sdt,
                sdt2,
                data.strategy);
            data.x.col(i) = ToEigen(x);
        });
        // Minimize Backward Euler, i.e. BDF1, objective
        Smoother S{iterations};
        S.Apply(sdt, rho, data);
        // Update velocity
        data.vt = data.v;
        tbb::parallel_for(Index(0), nVertices, [&](Index i) {
            auto v = kernels::IntegrateVelocity(
                FromEigen(data.xt.col(i).head<3>()),
                FromEigen(data.x.col(i).head<3>()),
                sdt);
            data.v.col(i) = ToEigen(v);
        });
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/common/Eigen.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/physics/HyperElasticity.h"

#include <doctest/doctest.h>
#include <span>

TEST_CASE("[sim][vbd] Integrator")
{
    using namespace pbat;
    // Arrange
    // Cube mesh
    MatrixX P(3, 8);
    IndexMatrixX V(1, 8);
    IndexMatrixX T(4, 5);
    IndexMatrixX F(3, 12);
    // clang-format off
    P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    F << 0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 4, 5,
         1, 5, 3, 7, 2, 6, 0, 4, 3, 2, 5, 7,
         4, 4, 5, 5, 7, 7, 6, 6, 1, 3, 6, 6;
    // clang-format on
    V.reshaped().setLinSpaced(0, static_cast<Index>(P.cols() - 1));
    // Problem parameters
    auto constexpr dt         = Scalar{1e-2};
    auto constexpr substeps   = 1;
    auto constexpr iterations = 10;

    // Act
    using pbat::common::ToEigen;
    using pbat::sim::vbd::Integrator;
    Integrator vbd{sim::vbd::Data().WithVolumeMesh(P, T).WithSurfaceMesh(V, T).Construct()};
    vbd.Step(dt, iterations, substeps);

    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = vbd.data.x - P;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}