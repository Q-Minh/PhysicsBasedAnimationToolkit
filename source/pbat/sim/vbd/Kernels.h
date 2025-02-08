#ifndef PBAT_SIM_VBD_KERNELS_H
#define PBAT_SIM_VBD_KERNELS_H

#include "Enums.h"
#include "pbat/HostDevice.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/DeformationGradient.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/IntersectionQueries.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/HyperElasticity.h"

#include <cmath>
#include <limits>

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

/**
 * @brief
 *
 * @tparam TMatrixXTV
 * @tparam TMatrixXV
 * @tparam TMatrixXTF
 * @tparam TMatrixXF
 * @tparam TMatrixG
 * @tparam TMatrixH
 * @tparam TMatrixXV::ScalarType
 * @param xtv 3x1 vertex positions at time t
 * @param xv 3x1 vertex positions
 * @param xtf 3x1 triangle positions at time t
 * @param xf 3x1 triangle positions
 * @param dt Time step
 * @param muC Collision penalty
 * @param muF Friction coefficient
 * @param epsv IPC's relative velocity threshold for static to dynamic friction's smooth transition
 * @param g Vertex gradient
 * @param H Vertex hessian
 * @return
 */
template <
    mini::CMatrix TMatrixXTV,
    mini::CMatrix TMatrixXV,
    mini::CMatrix TMatrixXTF,
    mini::CMatrix TMatrixXF,
    mini::CMatrix TMatrixG,
    mini::CMatrix TMatrixH,
    class ScalarType = typename TMatrixXV::ScalarType>
PBAT_HOST_DEVICE void AccumulateVertexTriangleContact(
    TMatrixXTV const& xtv,
    TMatrixXV const& xv,
    TMatrixXTF const& xtf,
    TMatrixXF const& xf,
    ScalarType dt,
    ScalarType muC,
    ScalarType muF,
    ScalarType epsv,
    TMatrixG& g,
    TMatrixH& H)
{
    using namespace mini;
    // Compute triangle normal
    SMatrix<ScalarType, 3, 2> T{};
    T.Col(0)                         = xf.Col(1) - xf.Col(0);
    T.Col(1)                         = xf.Col(2) - xf.Col(0);
    SVector<ScalarType, 3> n         = Cross(T.Col(0), T.Col(1));
    ScalarType const doublearea      = Norm(n);
    bool const bIsTriangleDegenerate = doublearea <= ScalarType(1e-8);
    if (bIsTriangleDegenerate)
        return;

    n /= doublearea;
    using namespace pbat::geometry;
    SVector<ScalarType, 3> xc = ClosestPointQueries::PointOnPlane(xv, xf.Col(0), n);
    // Check if xv projects to the triangle's interior by checking its barycentric coordinates
    SVector<ScalarType, 3> b =
        IntersectionQueries::TriangleBarycentricCoordinates(xc - xf.Col(0), T.Col(0), T.Col(1));
    // If xv doesn't project inside triangle, then we don't generate a contact response
    // clang-format off
    bool const bIsVertexInsideTriangle = All(b >= ScalarType(0) and b <= ScalarType(1));
    // clang-format on
    if (not bIsVertexInsideTriangle)
        return;

    // Collision energy is \frac{1}{2} \mu_C [(xv - xb)^T n]^2
    SVector<ScalarType, 3> xb = xf * b;
    ScalarType d              = min(ScalarType(0), Dot(xv - xb, n));
    ScalarType lambda         = muC * d;
    // Gradient is muC [(xv - xb)^T n] I_{d x d} n
    g += lambda * n;
    // Hessian is muC n n^T
    H += muC * (n * n.Transpose());

    // IPC smooth friction energy is \mu_F
    T.Col(1)                 = Cross(n, T.Col(0)); ///< Binormal
    auto xtb                 = xtf * b;
    auto dx                  = (xv - xtv) - (xb - xtb);
    SVector<ScalarType, 2> u = T.Transpose() * dx;
    ScalarType unorm         = Norm(u) + std::numeric_limits<ScalarType>::epsilon();
    ScalarType uepsvh        = unorm / (epsv * dt);
    ScalarType f1            = (uepsvh < 1) ? 2 * uepsvh - (uepsvh * uepsvh) : ScalarType(1);
    // Gradient is \mu_F \lambda T f1 \frac{u}{\norm{u}}
    ScalarType muFlambdaf1unorm = (muF * abs(lambda) * f1) / unorm;
    g += muFlambdaf1unorm * T * u;
    H += muFlambdaf1unorm * T * T.Transpose();
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

} // namespace kernels
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_KERNELS_H
