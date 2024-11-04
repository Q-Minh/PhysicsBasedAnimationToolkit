#include "Integrator.h"

#include "Kernels.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <tbb/parallel_for.h>
#include <type_traits>

namespace pbat {
namespace sim {
namespace vbd {

Integrator::Integrator(Data&& dataIn) : data(dataIn) {}

void Integrator::Step(Scalar dt, Index iterations, Index substeps, Scalar rho)
{
    Scalar sdt                           = dt / (static_cast<Scalar>(substeps));
    Scalar sdt2                          = sdt * sdt;
    auto const nVertices                 = data.x.cols();
    using IndexType                      = std::remove_const_t<decltype(nVertices)>;
    bool const bUseChebyshevAcceleration = rho > Scalar(0) and rho < Scalar(1);
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::ToEigen;
    for (auto s = 0; s < substeps; ++s)
    {
        // Store previous positions
        data.xt = data.x;
        // Compute inertial target positions
        tbb::parallel_for(IndexType(0), nVertices, [&](auto i) {
            auto xtilde = kernels::InertialTarget(
                FromEigen(data.xt.col(i).head<3>()),
                FromEigen(data.vt.col(i).head<3>()),
                FromEigen(data.aext.col(i).head<3>()),
                sdt,
                sdt2);
            data.xtilde.col(i) = ToEigen(xtilde);
        });
        // Initialize block coordinate descent's, i.e. BCD's, solution
        tbb::parallel_for(IndexType(0), nVertices, [&](auto i) {
            auto x = kernels::InitialPositionsForSolve(
                FromEigen(data.xt.col(i).head<3>()),
                FromEigen(data.vt.col(i).head<3>()),
                FromEigen(data.v.col(i).head<3>()),
                FromEigen(data.aext.col(i).head<3>()),
                sdt,
                sdt2,
                data.initializationStrategy);
            data.x.col(i) = ToEigen(x);
        });
        // Initialize Chebyshev semi-iterative method
        Scalar omega{};
        Scalar rho2 = rho * rho;
        // Minimize Backward Euler, i.e. BDF1, objective
        for (auto k = 0; k < iterations; ++k)
        {
            if (bUseChebyshevAcceleration)
                omega = kernels::ChebyshevOmega(k, rho2, omega);

            for (auto const& partition : data.partitions)
            {
                auto const nVerticesInPartition = static_cast<std::size_t>(partition.size());
                tbb::parallel_for(std::size_t(0), nVerticesInPartition, [&](auto v) {
                    auto i     = partition[v];
                    auto begin = data.GVGp[i];
                    auto end   = data.GVGp[i + 1];
                    // Compute vertex elastic hessian
                    mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                    mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                    for (auto n = begin; n < end; ++n)
                    {
                        auto ilocal                     = data.GVGilocal[n];
                        auto e                          = data.GVGe[n];
                        auto g                          = data.GVGn[n];
                        auto lamee                      = data.lame.col(g);
                        auto wg                         = data.wg[g];
                        auto Te                         = data.T.col(e);
                        mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(data.GP.block<4, 3>(0, e * 3));
                        mini::SMatrix<Scalar, 3, 4> xe =
                            FromEigen(data.x(Eigen::all, Te).block<3, 4>(0, 0));
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
                tbb::parallel_for(IndexType(0), nVertices, [&](auto i) {
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
        tbb::parallel_for(IndexType(0), nVertices, [&](auto i) {
            kernels::IntegrateVelocity(
                FromEigen(data.xt.col(i).head<3>()),
                FromEigen(data.x.col(i).head<3>()),
                sdt);
        });
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat