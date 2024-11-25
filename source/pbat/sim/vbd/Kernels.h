#ifndef PBAT_SIM_VBD_KERNELS_H
#define PBAT_SIM_VBD_KERNELS_H

#include "Enums.h"
#include "pbat/HostDevice.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/DeformationGradient.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/HyperElasticity.h"

#include <cmath>

namespace pbat {
namespace sim {
namespace vbd {
namespace kernels {

namespace mini = math::linalg::mini;

template <
    mini::CMatrix TMatrixXT,
    mini::CMatrix TMatrixVT,
    mini::CMatrix TMatrixA,
    class ScalarType = typename TMatrixXT::ScalarType>
PBAT_HOST_DEVICE mini::SVector<ScalarType, TMatrixXT::kRows> InertialTarget(
    TMatrixXT const& xt,
    TMatrixVT const& vt,
    TMatrixA const& aext,
    ScalarType dt,
    ScalarType dt2)
{
    return xt + dt * vt + dt2 * aext;
}

template <
    mini::CMatrix TMatrixXT,
    mini::CMatrix TMatrixVTM1,
    mini::CMatrix TMatrixVT,
    mini::CMatrix TMatrixA,
    class ScalarType = typename TMatrixXT::ScalarType>
PBAT_HOST_DEVICE mini::SVector<ScalarType, TMatrixXT::kRows> InitialPositionsForSolve(
    TMatrixXT const& xt,
    TMatrixVTM1 const& vtm1,
    TMatrixVT const& vt,
    TMatrixA const& aext,
    ScalarType dt,
    ScalarType dt2,
    EInitializationStrategy strategy)
{
    using namespace mini;
    if (strategy == EInitializationStrategy::Position)
    {
        return xt;
    }
    else if (strategy == EInitializationStrategy::Inertia)
    {
        return xt + dt * vt;
    }
    else if (strategy == EInitializationStrategy::KineticEnergyMinimum)
    {
        return xt + dt * vt + dt2 * aext;
    }
    else // (strategy == EInitializationStrategy::AdaptiveVbd)
    {
        ScalarType const aextn2                 = SquaredNorm(aext);
        bool const bHasZeroExternalAcceleration = (aextn2 == ScalarType(0));
        ScalarType atilde{0};
        if (not bHasZeroExternalAcceleration)
        {
            using namespace std;
            auto constexpr kRows = TMatrixXT::kRows;
            if (strategy == EInitializationStrategy::AdaptiveVbd)
            {
                SVector<ScalarType, kRows> const ati = (vt - vtm1) / dt;
                atilde                               = Dot(ati, aext) / aextn2;
                atilde = min(max(atilde, ScalarType(0)), ScalarType(1));
            }
            if (strategy == EInitializationStrategy::AdaptivePbat)
            {
                SVector<ScalarType, kRows> const dti =
                    vt / (Norm(vt) + std::numeric_limits<ScalarType>::min());
                atilde = Dot(dti, aext) / aextn2;
                // Discard the sign of atilde, because motion that goes against
                // gravity should "feel" gravity, rather than ignore it (i.e. clamping).
                atilde = min(abs(atilde), ScalarType(1));
            }
        }
        return xt + dt * vt + dt2 * atilde * aext;
    }
}

template <class ScalarType, class IndexType>
PBAT_HOST ScalarType ChebyshevOmega(IndexType k, ScalarType rho2, ScalarType omega = {})
{
    return (k == IndexType(0)) ? ScalarType{1} :
           (k == IndexType(1)) ? ScalarType{2} / (ScalarType{2} - rho2) :
                                 ScalarType{4} / ScalarType{4} - rho2 * omega;
}

template <
    mini::CMatrix TMatrixXKM2,
    mini::CMatrix TMatrixXKM1,
    mini::CMatrix TMatrixXK,
    class IndexType,
    class ScalarType = typename TMatrixXK::ScalarType>
PBAT_HOST_DEVICE void
ChebyshevUpdate(IndexType k, ScalarType omega, TMatrixXKM2& xkm2, TMatrixXKM1& xkm1, TMatrixXK& xk)
{
    if (k > 1)
    {
        xk = omega * (xk - xkm2) + xkm2;
    }
    xkm2 = xkm1;
    xkm1 = xk;
}

template <
    mini::CMatrix TMatrixXT,
    mini::CMatrix TMatrixX,
    class ScalarType = typename TMatrixXT::ScalarType>
PBAT_HOST_DEVICE mini::SVector<ScalarType, TMatrixXT::kRows>
IntegrateVelocity(TMatrixXT const& xt, TMatrixX const& x, ScalarType dt)
{
    return (x - xt) / dt;
}

template <
    mini::CMatrix TMatrixGP,
    mini::CMatrix TMatrixHF,
    mini::CMatrix TMatrixHI,
    class IndexType,
    class ScalarType = typename TMatrixGP::ScalarType>
PBAT_HOST_DEVICE void AccumulateElasticHessian(
    IndexType ilocal,
    ScalarType wg,
    TMatrixGP const& GP,
    TMatrixHF const& HF,
    TMatrixHI& Hi)
{
    auto constexpr kDims = TMatrixGP::kCols;
    // Contract (d^k Psi / dF^k) with (d F / dx)^k. See pbat/fem/DeformationGradient.h.
    common::ForRange<0, kDims>([&]<auto kj>() {
        common::ForRange<0, kDims>([&]<auto ki>() {
            Hi += wg * GP(ilocal, ki) * GP(ilocal, kj) *
                  HF.template Slice<kDims, kDims>(ki * kDims, kj * kDims);
        });
    });
}

template <
    mini::CMatrix TMatrixGP,
    mini::CMatrix TMatrixGF,
    mini::CMatrix TMatrixGI,
    class IndexType,
    class ScalarType = typename TMatrixGP::ScalarType>
PBAT_HOST_DEVICE void AccumulateElasticGradient(
    IndexType ilocal,
    ScalarType wg,
    TMatrixGP const& GP,
    TMatrixGF const& gF,
    TMatrixGI& gi)
{
    auto constexpr kDims = TMatrixGP::kCols;
    // Contract (d^k Psi / dF^k) with (d F / dx)^k. See pbat/fem/DeformationGradient.h.
    common::ForRange<0, kDims>(
        [&]<auto k>() { gi += wg * GP(ilocal, k) * gF.template Slice<kDims, 1>(k * kDims, 0); });
}

template <
    mini::CMatrix TMatrixXT,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixG,
    mini::CMatrix TMatrixH,
    class ScalarType = typename TMatrixXT::ScalarType>
PBAT_HOST_DEVICE void AddDamping(
    ScalarType dt,
    TMatrixXT const& xt,
    TMatrixX const& x,
    ScalarType kD,
    TMatrixG& g,
    TMatrixH& H)
{
    // Add Rayleigh damping terms
    ScalarType const D = kD / dt;
    g += D * (H * (x - xt));
    H *= ScalarType{1} + D;
}

template <
    mini::CMatrix TMatrixXTL,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixG,
    mini::CMatrix TMatrixH,
    class ScalarType = typename TMatrixXTL::ScalarType>
PBAT_HOST_DEVICE void AddInertiaDerivatives(
    ScalarType dt2,
    ScalarType m,
    TMatrixXTL const& xtilde,
    TMatrixX const& x,
    TMatrixG& g,
    TMatrixH& H)
{
    // Add inertial energy derivatives
    ScalarType const K = m / dt2;
    Diag(H) += K;
    g += K * (x - xtilde);
}

template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixG,
    mini::CMatrix TMatrixH,
    class ScalarType = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE void IntegratePositions(
    TMatrixG const& g,
    TMatrixH const& H,
    TMatrixX& x,
    ScalarType detHZero = ScalarType(1e-7))
{
    // 3. Newton step
    if (abs(Determinant(H)) <= detHZero) // Skip nearly rank-deficient hessian
        return;
    x -= (Inverse(H) * g);
}

namespace restriction {

template <
    class IndexType,
    physics::CHyperElasticEnergy TPsi,
    mini::CMatrix TMatrixXCG,
    mini::CMatrix TMatrixGNCG,
    mini::CMatrix TMatrixGI,
    mini::CMatrix TMatrixHI,
    class ScalarType = typename TMatrixXCG::ScalarType>
void AccumulateSingularEnergy(
    IndexType ilocal,
    ScalarType wg,
    TPsi const& Psi,
    ScalarType mug,
    ScalarType lambdag,
    TMatrixXCG const& xcg,
    TMatrixGNCG const& GNcg,
    TMatrixGI& gi,
    TMatrixHI& Hi)
{
    using namespace mini;
    SMatrix<Scalar, 3, 3> F = xcg * GNcg;
    SVector<Scalar, 9> gF;
    SMatrix<Scalar, 9, 9> HF;
    Psi.gradAndHessian(F, mug, lambdag, gF, HF);
    kernels::AccumulateElasticGradient(ilocal, wg, GNcg, gF, gi);
    kernels::AccumulateElasticHessian(ilocal, wg, GNcg, HF, Hi);
}

template <
    class IndexType,
    mini::CMatrix TMatrixXCG,
    mini::CMatrix TMatrixNCG,
    mini::CMatrix TMatrixXTL,
    mini::CMatrix TMatrixGI,
    mini::CMatrix TMatrixHI,
    class ScalarType = typename TMatrixXCG::ScalarType>
void AccumulateShapeMatchingEnergy(
    IndexType ilocal,
    ScalarType wg,
    ScalarType rhog,
    TMatrixXCG const& xcg,
    TMatrixNCG const& Ncg,
    TMatrixXTL const& xf,
    TMatrixGI& gi,
    TMatrixHI& Hi)
{
    using namespace mini;
    auto xc = xcg * Ncg;
    // Energy is 1/2 w_g rho_g || xc - xf ||_2^2
    gi += (wg * rhog * Ncg(ilocal)) * (xc - xf);
    Diag(Hi) += wg * rhog * Ncg(ilocal) * Ncg(ilocal);
}

} // namespace restriction

namespace smoothing {

template <
    class IndexType,
    mini::CMatrix TMatrixXCG,
    mini::CMatrix TMatrixNC,
    mini::CMatrix TMatrixXTL,
    mini::CMatrix TMatrixGI,
    mini::CMatrix TMatrixHI,
    class ScalarType = typename TMatrixXCG::ScalarType>
void AccumulateKineticEnergy(
    IndexType ilocal,
    ScalarType wg,
    ScalarType rhog,
    TMatrixXCG const& xcg,
    TMatrixNC const& Nc,
    TMatrixXTL const& xtildeg,
    TMatrixGI& gi,
    TMatrixHI& Hi)
{
    using namespace mini;
    // Kinetic energy is 1/2 w_g rho_g || Nc*xc - xtildeg ||_2^2
    auto x = xcg * Nc;
    gi += wg * rhog * (x - xtildeg) * Nc(ilocal);
    Diag(Hi) += wg * rhog * Nc(ilocal) * Nc(ilocal);
}

template <
    class IndexType,
    physics::CHyperElasticEnergy TPsi,
    mini::CMatrix TMatrixXCG,
    mini::CMatrix TMatrixGNCG,
    mini::CMatrix TMatrixGI,
    mini::CMatrix TMatrixHI,
    class ScalarType = typename TMatrixXCG::ScalarType>
void AccumulateSingularEnergy(
    ScalarType dt2,
    IndexType ilocal,
    ScalarType wg,
    TPsi const& Psi,
    ScalarType mug,
    ScalarType lambdag,
    TMatrixXCG const& xcg,
    TMatrixGNCG const& GNcg,
    TMatrixGI& gi,
    TMatrixHI& Hi)
{
    using namespace mini;
    SMatrix<Scalar, 3, 3> F = xcg * GNcg;
    SVector<Scalar, 9> gF;
    SMatrix<Scalar, 9, 9> HF;
    Psi.gradAndHessian(F, mug, lambdag, gF, HF);
    kernels::AccumulateElasticGradient(ilocal, dt2 * wg, GNcg, gF, gi);
    kernels::AccumulateElasticHessian(ilocal, dt2 * wg, GNcg, HF, Hi);
}

template <
    class IndexType,
    mini::CMatrix TMatrixI,
    mini::CMatrix TMatrixNCE,
    mini::CMatrix TMatrixGE,
    mini::CMatrix TMatrixGI,
    class ScalarType = typename TMatrixNCE::ScalarType>
void AccumulateElasticGradient(
    IndexType ilocal,
    ScalarType wg,
    TMatrixI const& indicator,
    TMatrixNCE const& Nce,
    TMatrixGE const& ge,
    TMatrixGI& gi)
{
    common::ForRange<0, 4>(
        [&]<auto k>() { gi += indicator(k) * wg * Nce(ilocal, k) * ge.Slice<3, 1>(k * 3, 0); });
}

template <
    class IndexType,
    mini::CMatrix TMatrixI,
    mini::CMatrix TMatrixNCE,
    mini::CMatrix TMatrixHE,
    mini::CMatrix TMatrixHI,
    class ScalarType = typename TMatrixNCE::ScalarType>
void AccumulateElasticHessian(
    IndexType ilocal,
    ScalarType wg,
    TMatrixI const& indicator,
    TMatrixNCE const& Nce,
    TMatrixHE const& He,
    TMatrixHI& Hi)
{
    common::ForRange<0, 4>([&]<auto kj>() {
        common::ForRange<0, 4>([&]<auto ki>() {
            Hi += indicator(ki) * indicator(kj) * wg * Nce(ilocal, ki) * Nce(ilocal, kj) *
                  He.Slice<3, 3>(ki * 3, kj * 3);
        });
    });
}

template <
    mini::CMatrix TMatrixXCR,
    mini::CMatrix TMatrixNCE,
    mini::CMatrix TMatrixGN,
    class ScalarType = typename TMatrixXCR::ScalarType>
mini::SMatrix<ScalarType, 3, 3>
ComputeDeformationGradient(TMatrixXCR const& xcr, TMatrixNCE const& Nce, TMatrixGN const& GN)
{
    using namespace mini;
    SMatrix<Scalar, 3, 4> x{};
    common::ForRange<0, 4>(
        [&]<auto k>() { x.Col(k) = xcr.template Slice<3, 4>(0, k * 4) * Nce.Col(k); });
    auto F = x * GN;
    return F;
}

template <
    physics::CHyperElasticEnergy TPsi,
    mini::CMatrix TMatrixXCR,
    mini::CMatrix TMatrixNCE,
    mini::CMatrix TMatrixGN,
    mini::CMatrix TMatrixGF,
    mini::CMatrix TMatrixHF,
    class ScalarType = typename TMatrixXCR::ScalarType>
void ComputeElasticDerivativesWrtF(
    TPsi const& Psi,
    ScalarType mug,
    ScalarType lambdag,
    TMatrixXCR const& xcr,
    TMatrixNCE const& Nce,
    TMatrixGN const& GN,
    TMatrixGF& gF,
    TMatrixHF& HF)
{
    using namespace mini;
    SMatrix<Scalar, 3, 3> F = ComputeDeformationGradient(xcr, Nce, GN);
    Psi.gradAndHessian(F, mug, lambdag, gF, HF);
}

template <
    class IndexType,
    physics::CHyperElasticEnergy TPsi,
    mini::CMatrix TMatrixXCR,
    mini::CMatrix TMatrixNCE,
    mini::CMatrix TMatrixGN,
    mini::CMatrix TMatrixI,
    mini::CMatrix TMatrixGI,
    mini::CMatrix TMatrixHI,
    class ScalarType = typename TMatrixXCR::ScalarType>
void AccumulatePotentialEnergy(
    ScalarType dt2,
    IndexType ilocal,
    ScalarType wg,
    TPsi const& Psi,
    ScalarType mug,
    ScalarType lambdag,
    TMatrixXCR const& xcr,
    TMatrixNCE const& Nce,
    TMatrixGN const& GN,
    TMatrixI const& indicator,
    TMatrixGI& gi,
    TMatrixHI& Hi)
{
    using namespace mini;
    // Potential energy is w_g \Psi(Nce * xcr)
    SVector<Scalar, 9> gF;
    SMatrix<Scalar, 9, 9> HF;
    ComputeElasticDerivativesWrtF(Psi, mug, lambdag, xcr, Nce, GN, gF, HF);
    using ElementType          = fem::Tetrahedron<1>;
    SVector<Scalar, 12> ge     = fem::GradientWrtDofs<ElementType, 3>(gF, GN);
    SMatrix<Scalar, 12, 12> He = fem::HessianWrtDofs<ElementType, 3>(HF, GN);
    AccumulateElasticGradient(ilocal, dt2 * wg, indicator, Nce, ge, gi);
    AccumulateElasticHessian(ilocal, dt2 * wg, indicator, Nce, He, Hi);
}

} // namespace smoothing

} // namespace kernels
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_KERNELS_H