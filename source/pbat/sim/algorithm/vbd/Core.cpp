#include "Core.h"

#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/algorithm/vbd/Kernels.h"

#include <exception>
#include <fmt/core.h>
#include <tbb/parallel_for.h>

namespace pbat::sim::algorithm::vbd {

PBAT_API Params& Params::WithVertexElementAdjacencyGraph(
    Eigen::Ref<IndexVectorX const> const& _GVGp,
    Eigen::Ref<IndexVectorX const> const& _GVGe,
    Eigen::Ref<IndexVectorX const> const& _GVGilocal)
{
    GVGp      = _GVGp;
    GVGe      = _GVGe;
    GVGilocal = _GVGilocal;
    return *this;
}

PBAT_API Params& Params::WithVertexColors(Eigen::Ref<IndexVectorX const> const& _colors)
{
    colors = _colors;
    return *this;
}

PBAT_API Params& Params::WithInitializationStrategy(EInitializationStrategy _strategy)
{
    strategy = _strategy;
    return *this;
}

PBAT_API Params& Params::WithHessianDeterminantZeroUnder(Scalar zero)
{
    detHZero = zero;
    return *this;
}

PBAT_API Params& Params::Construct(bool bValidate)
{
    if (bValidate)
    {
        auto nVerts = colors.size();
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
    }
    return *this;
}

void InitializeSolve(FemElastoDynamics& fem, Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.InitializeSolve");
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::ToEigen;
    // NOTE:
    // We should make this initialization adapt to higher-order BDF schemes as well!
    // In this case, we would have
    // "xt" = -fem.bdf.Inertia(0), and
    // "vt" = -fem.bdf.Inertia(1),
    // and instead of the h, we would have fem.bdf.BetaTilde()
    // and h^2 would be fem.bdf.BetaTilde()^2.
    auto aext             = fem.aext();
    auto xt               = fem.bdf.CurrentState(0).reshaped(fem.x.rows(), fem.x.cols());
    auto vt               = fem.bdf.CurrentState(1).reshaped(fem.v.rows(), fem.v.cols());
    auto free             = fem.FreeNodes();
    auto const nFreeVerts = free.size();
    auto h                = fem.bdf.TimeStep();
    auto h2               = h * h;
    tbb::parallel_for(Index(0), nFreeVerts, [&](Index fi) {
        auto i = free(fi);
        auto x = kernels::InitialPositionsForSolve(
            FromEigen(xt.col(i).head<3>()),
            FromEigen(vt.col(i).head<3>()),
            FromEigen(fem.v.col(i).head<3>()),
            FromEigen(aext.col(i).head<3>()),
            h,
            h2,
            params.strategy);
        fem.x.col(i) = ToEigen(x);
    });
}

void Step(FemElastoDynamics& fem, Params const& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Step");
    auto h                 = fem.bdf.TimeStep();
    auto h2                = h * h;
    auto xt                = fem.bdf.CurrentState(0).reshaped(fem.x.rows(), fem.x.cols());
    auto vt                = fem.bdf.CurrentState(1).reshaped(fem.v.rows(), fem.v.cols());
    auto const nPartitions = params.Pptr.size() - 1;
    for (Index p = 0; p < nPartitions; ++p)
    {
        auto const pBegin = params.Pptr(p);
        auto const pEnd   = params.Pptr(p + 1);
        tbb::parallel_for(pBegin, pEnd, [&](Index k) {
            // Solve vertex i
            using namespace math::linalg;
            using mini::FromEigen;
            using mini::ToEigen;
            auto i = params.Padj(k);
            if (fem.IsDirichletNode(i))
                return;
            auto begin = params.GVGp(i);
            auto end   = params.GVGp(i + 1);
            // Elastic energy
            mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
            mini::SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
            for (auto n = begin; n < end; ++n)
            {
                auto ilocal                     = params.GVGilocal(n);
                auto e                          = params.GVGe(n);
                auto lamee                      = fem.lamegU.col(e);
                auto wg                         = fem.wgU(e);
                auto ti                         = fem.mesh.E.col(e);
                mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(fem.GNegU.block<4, 3>(0, e * 3));
                mini::SMatrix<Scalar, 3, 4> xe =
                    FromEigen(fem.x(Eigen::placeholders::all, ti).block<3, 4>(0, 0));
                mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
                physics::StableNeoHookeanEnergy<3> Psi{};
                mini::SVector<Scalar, 9> gF;
                mini::SMatrix<Scalar, 9, 9> HF;
                Psi.gradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
                kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
                kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
            }
            // "Kinetic" energy
            Scalar m                         = fem.m(i);
            mini::SVector<Scalar, 3> xti     = FromEigen(xt.col(i).head<3>());
            mini::SVector<Scalar, 3> xtildei = FromEigen(fem.xtilde.col(i).head<3>());
            mini::SVector<Scalar, 3> xi      = FromEigen(fem.x.col(i).head<3>());
            kernels::AddDamping(h, xti, xi, Scalar(0) /*Rayleigh damping*/, gi, Hi);
            kernels::AddInertiaDerivatives(h2, m, xtildei, xi, gi, Hi);
            // Update vertex position
            kernels::IntegratePositions(gi, Hi, xi, params.detHZero);
            fem.x.col(i) = ToEigen(xi);
        });
    }
}

} // namespace pbat::sim::algorithm::vbd