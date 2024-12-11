#include "Integrator.h"

#include "Kernels.h"
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
        bool const bUseChebyshevAcceleration = rho > Scalar(0) and rho < Scalar(1);
        Scalar omega{};
        Scalar rho2 = rho * rho;
        // Minimize Backward Euler, i.e. BDF1, objective
        for (auto k = 0; k < iterations; ++k)
        {
            if (bUseChebyshevAcceleration)
                omega = kernels::ChebyshevOmega(k, rho2, omega);

            auto const nPartitions = data.Pptr.size() - 1;
            for (auto p = 0; p < nPartitions; ++p)
            {
                auto const pBegin = data.Pptr(p);
                auto const pEnd   = data.Pptr(p + 1);
                tbb::parallel_for(pBegin, pEnd, [&](Index k) {
                    using namespace math::linalg;
                    using mini::FromEigen;
                    using mini::ToEigen;

                    auto i     = data.Padj[k];
                    auto begin = data.GVGp(i);
                    auto end   = data.GVGp(i + 1);
                    // Elastic energy
                    mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                    mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                    for (auto n = begin; n < end; ++n)
                    {
                        auto ilocal                     = data.GVGilocal[n];
                        auto e                          = data.GVGe[n];
                        auto lamee                      = data.lame.col(e);
                        auto wg                         = data.wg[e];
                        auto Te                         = data.mesh.E.col(e);
                        mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(data.GP.block<4, 3>(0, e * 3));
                        mini::SMatrix<Scalar, 3, 4> xe =
                            FromEigen(data.x(Eigen::placeholders::all, Te).block<3, 4>(0, 0));
                        mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
                        physics::StableNeoHookeanEnergy<3> Psi{};
                        mini::SVector<Scalar, 9> gF;
                        mini::SMatrix<Scalar, 9, 9> HF;
                        Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
                        kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
                        kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
                    }
                    // Update vertex position
                    Scalar m                         = data.m[i];
                    mini::SVector<Scalar, 3> xti     = FromEigen(data.xt.col(i).head<3>());
                    mini::SVector<Scalar, 3> xtildei = FromEigen(data.xtilde.col(i).head<3>());
                    mini::SVector<Scalar, 3> xi      = FromEigen(data.x.col(i).head<3>());
                    kernels::AddDamping(sdt, xti, xi, data.kD, gi, Hi);
                    kernels::AddInertiaDerivatives(sdt2, m, xtildei, xi, gi, Hi);
                    kernels::IntegratePositions(gi, Hi, xi, data.detHZero);
                    data.x.col(i) = ToEigen(xi);
                });
            }

            if (bUseChebyshevAcceleration)
            {
                tbb::parallel_for(Index(0), nVertices, [&](Index i) {
                    using namespace math::linalg;
                    using mini::FromEigen;
                    auto xkm2eig = data.xchebm2.col(i).head<3>();
                    auto xkm1eig = data.xchebm1.col(i).head<3>();
                    auto xkeig   = data.x.col(i).head<3>();
                    auto xkm2    = FromEigen(xkm2eig);
                    auto xkm1    = FromEigen(xkm1eig);
                    auto xk      = FromEigen(xkeig);
                    kernels::ChebyshevUpdate(k, omega, xkm2, xkm1, xk);
                });
            }
        }
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