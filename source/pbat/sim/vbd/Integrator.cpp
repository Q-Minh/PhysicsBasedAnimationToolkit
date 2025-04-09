#include "Integrator.h"

#include "Kernels.h"
#include "pbat/common/Eigen.h"
#include "pbat/fem/DeformationGradient.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/profiling/Profiling.h"

#include <fmt/format.h>
#include <tbb/parallel_for.h>
#include <type_traits>
#include <unsupported/Eigen/SparseExtra>

namespace pbat {
namespace sim {
namespace vbd {

Integrator::Integrator(Data dataIn) : data(std::move(dataIn)) {}

void Integrator::Step(Scalar dt, Index iterations, Index substeps)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.Integrator.Step");

    Scalar sdt  = dt / (static_cast<Scalar>(substeps));
    Scalar sdt2 = sdt * sdt;
    for (auto s = 0; s < substeps; ++s)
    {
        // Store previous positions
        data.xt = data.x;
        data.vt = data.v;
        // Prepare optimization problem
        data.xtilde = data.xt + sdt * data.vt + sdt2 * data.aext;
        InitializeSolve(sdt, sdt2);
        // Minimize Backward Euler, i.e. BDF1, objective
        Solve(sdt, sdt2, iterations);
        // Update velocity
        data.v = (data.x - data.xt) / sdt;
        // Save descent path to disk if requested
        if (mTraceIterates)
            ExportTrace(sdt, s);
    }
    mTraceIterates = false;
}

PBAT_API void Integrator::TraceNextStep(std::string const& path, Index t)
{
    mTraceIterates = true;
    mTracePath     = path;
    mTimeStep      = t;
}

void Integrator::InitializeSolve(Scalar sdt, Scalar sdt2)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.Integrator.InitializeSolve");
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::ToEigen;
    auto const nVertices = data.x.cols();
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
}

void Integrator::RunVbdIteration(Scalar sdt, Scalar sdt2)
{
    TryTraceIteration(sdt);
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.Integrator.RunVbdIteration");
    auto const nPartitions = data.Pptr.size() - 1;
    for (Index p = 0; p < nPartitions; ++p)
    {
        auto const pBegin = data.Pptr(p);
        auto const pEnd   = data.Pptr(p + 1);
        tbb::parallel_for(pBegin, pEnd, [&](Index k) {
            auto i = data.Padj(k);
            SolveVertex(i, sdt, sdt2);
        });
    }
}

void Integrator::Solve(Scalar sdt, Scalar sdt2, Index iterations)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.Integrator.Solve");
    for (auto k = 0; k < iterations; ++k)
    {
        RunVbdIteration(sdt, sdt2);
    }
}

PBAT_API void Integrator::SolveVertex(Index i, Scalar sdt, Scalar sdt2)
{
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::ToEigen;

    auto begin = data.GVGp(i);
    auto end   = data.GVGp(i + 1);
    // Elastic energy
    mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
    mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
    for (auto n = begin; n < end; ++n)
    {
        auto ilocal                     = data.GVGilocal(n);
        auto e                          = data.GVGe(n);
        auto lamee                      = data.lame.col(e);
        auto wg                         = data.wg(e);
        auto ti                         = data.E.col(e);
        mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(data.GP.block<4, 3>(0, e * 3));
        mini::SMatrix<Scalar, 3, 4> xe =
            FromEigen(data.x(Eigen::placeholders::all, ti).block<3, 4>(0, 0));
        mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
        physics::StableNeoHookeanEnergy<3> Psi{};
        mini::SVector<Scalar, 9> gF;
        mini::SMatrix<Scalar, 9, 9> HF;
        Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
        kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
        kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
    }
    // Update vertex position
    Scalar m                         = data.m(i);
    mini::SVector<Scalar, 3> xti     = FromEigen(data.xt.col(i).head<3>());
    mini::SVector<Scalar, 3> xtildei = FromEigen(data.xtilde.col(i).head<3>());
    mini::SVector<Scalar, 3> xi      = FromEigen(data.x.col(i).head<3>());
    kernels::AddDamping(sdt, xti, xi, data.kD, gi, Hi);
    kernels::AddInertiaDerivatives(sdt2, m, xtildei, xi, gi, Hi);
    kernels::IntegratePositions(gi, Hi, xi, data.detHZero);
    data.x.col(i) = ToEigen(xi);
}

Scalar Integrator::ObjectiveFunction(
    Eigen::Ref<MatrixX const> const& xk,
    Eigen::Ref<MatrixX const> const& xtilde,
    Scalar dt)
{
    // Kinetic energy
    auto dx   = xk - xtilde;
    auto m    = data.m.replicate(1, 3).transpose();
    Scalar Ek = 0.5 * dx.reshaped().dot(m.cwiseProduct(dx).reshaped());
    // Elastic energy
    auto const nElements = data.E.cols();
    physics::StableNeoHookeanEnergy<3> Psi{};
    Scalar Ep{0};
    for (auto e = 0; e < nElements; ++e)
    {
        auto lamee = data.lame.col(e);
        auto wg    = data.wg(e);
        auto ti    = data.E.col(e);
        using namespace math::linalg;
        using mini::FromEigen;
        mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(data.GP.block<4, 3>(0, e * 3));
        mini::SMatrix<Scalar, 3, 4> xe =
            FromEigen(xk(Eigen::placeholders::all, ti).block<3, 4>(0, 0));
        mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
        Ep += wg * Psi.eval(Fe, lamee(0), lamee(1));
    }
    // Total energy
    return Ek + dt * dt * Ep;
}

VectorX Integrator::ObjectiveFunctionGradient(
    Eigen::Ref<MatrixX const> const& xk,
    Eigen::Ref<MatrixX const> const& xtilde,
    Scalar dt)
{
    // Kinetic energy
    auto dx     = xk - xtilde;
    auto m      = data.m.replicate(1, 3).transpose();
    VectorX gEk = m.cwiseProduct(dx).reshaped();
    // Elastic energy
    auto const nElements = data.E.cols();
    physics::StableNeoHookeanEnergy<3> Psi{};
    MatrixX gEp = MatrixX::Zero(data.x.rows(), data.x.cols());
    for (auto e = 0; e < nElements; ++e)
    {
        auto lamee = data.lame.col(e);
        auto wg    = data.wg(e);
        auto ti    = data.E.col(e);
        using namespace math::linalg;
        using mini::FromEigen;
        using mini::ToEigen;
        mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(data.GP.block<4, 3>(0, e * 3));
        mini::SMatrix<Scalar, 3, 4> xe =
            FromEigen(xk(Eigen::placeholders::all, ti).block<3, 4>(0, 0));
        mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
        mini::SVector<Scalar, 9> gF    = Psi.grad(Fe, lamee(0), lamee(1));
        using Element                  = fem::Tetrahedron<1>;
        auto ge                        = fem::GradientWrtDofs<Element, 3>(gF, GPe);
        gEp(Eigen::placeholders::all, ti) += wg * ToEigen(ge).reshaped(3, 4);
    }
    // Total energy
    return gEk + (dt * dt) * gEp.reshaped();
}

PBAT_API void Integrator::ExportTrace(Scalar sdt, Index substep)
{
    mTracedObjectives.push_back(ObjectiveFunction(data.x, data.xtilde, sdt));
    mTracedGradients.push_back(ObjectiveFunctionGradient(data.x, data.xtilde, sdt));
    mTracedPositions.push_back(data.x);
    auto const fPath    = fmt::format("{}/{}.{}.f.mtx", mTracePath, mTimeStep, substep);
    auto const gradPath = fmt::format("{}/{}.{}.grad.mtx", mTracePath, mTimeStep, substep);
    auto const xPath    = fmt::format("{}/{}.{}.x.mtx", mTracePath, mTimeStep, substep);
    Eigen::saveMarketDense(common::ToEigen(mTracedObjectives), fPath);
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    MatrixX G(mTracedGradients.front().size(), mTracedGradients.size());
    for (std::size_t i = 0; i < mTracedGradients.size(); ++i)
        G.col(i) = mTracedGradients[i];
    Eigen::saveMarketDense(G, gradPath);
    MatrixX XTR(mTracedPositions.front().size(), mTracedPositions.size());
    for (std::size_t i = 0; i < mTracedPositions.size(); ++i)
        XTR.col(i) = mTracedPositions[i].reshaped();
    Eigen::saveMarketDense(XTR, xPath);
#include "pbat/warning/Pop.h"
    mTracedObjectives.clear();
    mTracedGradients.clear();
    mTracedPositions.clear();
}

PBAT_API void Integrator::TryTraceIteration(Scalar sdt)
{
    if (mTraceIterates)
    {
        mTracedObjectives.push_back(ObjectiveFunction(data.x, data.xtilde, sdt));
        mTracedGradients.push_back(ObjectiveFunctionGradient(data.x, data.xtilde, sdt));
        mTracedPositions.push_back(data.x);
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/common/Eigen.h"

#include <doctest/doctest.h>

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
    P << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
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
    using pbat::common::ToEigen;
    using pbat::sim::vbd::Integrator;
    Integrator vbd{sim::vbd::Data().WithVolumeMesh(P, T).WithSurfaceMesh(V, F).Construct()};
    MatrixX xtilde = vbd.data.x + dt * vbd.data.v + dt * dt * vbd.data.aext;
    Scalar f0      = vbd.ObjectiveFunction(vbd.data.x, xtilde, dt);
    VectorX g0     = vbd.ObjectiveFunctionGradient(vbd.data.x, xtilde, dt);

    // Act
    vbd.Step(dt, iterations, substeps);

    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = vbd.data.x - P;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
    VectorX g    = vbd.ObjectiveFunctionGradient(vbd.data.x, xtilde, dt);
    Scalar gnorm = g.norm();
    CHECK_LT(gnorm, zero);
    Scalar f = vbd.ObjectiveFunction(vbd.data.x, xtilde, dt);
    CHECK_LT(f, f0);
}