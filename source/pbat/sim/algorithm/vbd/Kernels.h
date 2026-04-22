/**
 * @file Kernels.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Kernels for VBD algorithm.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_KERNELS_H
#define PBAT_SIM_ALGORITHM_VBD_KERNELS_H

#include "Enums.h"
#include "pbat/HostDevice.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/DeformationGradient.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/IntersectionQueries.h"
#include "pbat/math/linalg/FilterEigenvalues.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/sim/contact/Friction.h"
#include "pbat/sim/contact/Potentials.h"

#include <cmath>
#include <limits>

namespace pbat::sim::algorithm::vbd::kernels {

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
PBAT_HOST [[maybe_unused]] ScalarType ChebyshevOmega(IndexType k, ScalarType rho2, ScalarType omega)
{
    return (k == IndexType(0)) ? ScalarType{1} :
           (k == IndexType(1)) ? ScalarType{2} / (ScalarType{2} - rho2) :
                                 ScalarType{4} / (ScalarType{4} - rho2 * omega);
}

template <
    mini::CMatrix TMatrixXKM2,
    mini::CMatrix TMatrixXKM1,
    mini::CMatrix TMatrixXK,
    class IndexType,
    class ScalarType = typename TMatrixXK::ScalarType>
PBAT_HOST_DEVICE [[maybe_unused]] void
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
    pbat::common::ForRange<0, kDims>([&]<auto kj>() {
        pbat::common::ForRange<0, kDims>([&]<auto ki>() {
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
    pbat::common::ForRange<0, kDims>(
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
PBAT_HOST_DEVICE ScalarType AccumulateVertexTriangleContact(
    TMatrixXTV const& xtv,
    TMatrixXV const& xv,
    TMatrixXTF const& xtf,
    TMatrixXF const& xf,
    ScalarType dt,
    ScalarType muC,
    ScalarType muF,
    ScalarType epsv,
    TMatrixG* g = nullptr,
    TMatrixH* H = nullptr)
{
    using namespace mini;
    ScalarType E{0};

    // Compute triangle normal
    SMatrix<ScalarType, 3, 2> T{};
    T.Col(0)                         = xf.Col(1) - xf.Col(0);
    T.Col(1)                         = xf.Col(2) - xf.Col(0);
    SVector<ScalarType, 3> n         = Cross(T.Col(0), T.Col(1));
    ScalarType const doublearea      = Norm(n);
    bool const bIsTriangleDegenerate = doublearea <= ScalarType(1e-8);
    if (bIsTriangleDegenerate)
        return E;

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
        return E;

    // Collision energy is \f$ \frac{1}{2} \mu_C [(x_v - x_b)^T n]^2 \f$
    SVector<ScalarType, 3> xb = xf * b;
    ScalarType d              = min(ScalarType(0), Dot(xv - xb, n));
    ScalarType lambda         = muC * d;
    E += ScalarType(0.5) * lambda * d;
    // Gradient is \f$ \mu_C [(x_v - x_b)^T n] I_{d \times d} n \f$
    if (g)
        (*g) += lambda * n;
    // Hessian is \f$ \mu_C n n^T \f$
    if (H)
        (*H) += muC * (n * n.Transpose());

    // IPC smooth friction energy
    T.Col(1)                 = Cross(n, T.Col(0)); ///< Binormal
    auto xtb                 = xtf * b;
    auto dx                  = (xv - xtv) - (xb - xtb);
    SVector<ScalarType, 2> u = T.Transpose() * dx;
    ScalarType unorm         = Norm(u) + std::numeric_limits<ScalarType>::epsilon();
    ScalarType epsvh         = epsv * dt;
    ScalarType muFlambda     = muF * abs(lambda);
    ScalarType uepsvh        = unorm / epsvh;
    ScalarType f1            = (uepsvh < 1) ? 2 * uepsvh - (uepsvh * uepsvh) : ScalarType(1);
    // Gradient is \f$ \mu_F \lambda_N T f1(|u|) \frac{u}{|u|}} \f$
    ScalarType muFlambdaf1unorm = muFlambda * f1 / unorm;
    if (g)
        (*g) += muFlambdaf1unorm * T * u;
    if (H)
        (*H) += muFlambdaf1unorm * T * T.Transpose();
    // Energy is \f$ \mu_F \lambda_N f_0(|u|) \f$, where
    // \f$ f_0(y) = f_1'(y) \f$ and \f$ f_0(\eps_\nu h) = \eps_\nu h \f$
    // This yields \f$ f_0(y) = 1/3 \eps_\nu h + \frac{y^2}{\eps_\nu h} - \frac{y^3}{3 \eps_\nu^2
    // h^2} \f$
    ScalarType unorm2 = unorm * unorm;
    // clang-format off
    ScalarType f0     = 
        (ScalarType(1) / ScalarType(3) * epsvh) + 
        (unorm2 / epsvh) -
        (unorm2 * unorm / (ScalarType(3) * epsvh * epsvh));
    // clang-format on
    E += muFlambda * f0;
    return E;
}

/**
 * @brief Accumulate vertex to closest-point contact derivatives into gradient and hessian.
 *
 * @tparam TMatrixXI Type for vertex position
 * @tparam TMatrixXCP Type for closest point position
 * @tparam TMatrixXTI Type for vertex position at previous time step
 * @tparam TMatrixXTCP Type for closest point position at previous time step
 * @tparam TMatrixG Type for gradient
 * @tparam TMatrixH Type for hessian
 * @tparam ScalarType Scalar type
 * @param xi `3 x 1` vertex position
 * @param xj `3 x 1` closest point position
 * @param xti `3 x 1` vertex position at previous time step
 * @param xtcp `3 x 1` closest point position at previous time step
 * @param r Contact radius
 * @param kc Collision penalty
 * @param kcp `kcp = tau*kc*(tau - r)^2`, where `tau = r/2`
 * @param b `b = kc/2*(r - tau)^2 + kcp*log(tau)`, where `tau = r/2`
 * @param mu Friction coefficient
 * @param epsvh Time-step scaled relative velocity threshold for static to dynamic friction
 * transition
 * @param h2inv Inverse squared time step
 * @param g `3 x 1` gradient
 * @param H `3 x 3` hessian
 */
template <
    mini::CMatrix TMatrixXI,
    mini::CMatrix TMatrixXTI,
    mini::CMatrix TMatrixXCP,
    mini::CMatrix TMatrixXTCP,
    mini::CMatrix TMatrixG,
    mini::CMatrix TMatrixH,
    class ScalarType = typename TMatrixXI::ScalarType>
PBAT_HOST_DEVICE ScalarType AccumulateVertexClosestPointContactDerivatives(
    TMatrixXI const& xi,
    TMatrixXTI const& xti,
    TMatrixXCP const& xcp,
    TMatrixXTCP const& xtcp,
    ScalarType r,
    ScalarType kc,
    ScalarType kcp,
    ScalarType b,
    ScalarType mu,
    ScalarType epsvh,
    ScalarType h2inv,
    TMatrixG& g,
    TMatrixH& H)
{
    using namespace mini;
    // Normal contact
    ScalarType dij = Norm(xi - xcp);
    SVector<ScalarType, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(dij, r, kc, kcp, b);
    SVector<ScalarType, 3> dBdxi =
        contact::potentials::GradientSegmentWrtClosestPoints(xi, xcp, dij, dBdd(1), 0);
    SMatrix<ScalarType, 3, 3> d2Bdxi2 =
        contact::potentials::HessianBlockWrtClosestPoints(xi, xcp, dij, dBdd(1), dBdd(2), 0, 0);
    g += h2inv * dBdxi;
    H += h2inv * d2Bdxi2;
    // Frictional contact
    contact::potentials::LaggedFriction friction{};
    SMatrix<ScalarType, 3, 2> const T = contact::PointPointTangentialBasis(xi, xcp);
    SVector<ScalarType, 2> const uk   = T.Transpose() * ((xi - xti) - (xcp - xtcp));
    SVector<ScalarType, 2> gf;
    SMatrix<ScalarType, 2, 2> Hf;
    ScalarType lambda = -dBdd(1) * h2inv;
    friction.GradAndHessian(uk, mu, lambda, epsvh, gf, Hf);
    Hf = math::linalg::FilterEigenvalues(Hf, math::linalg::EEigenvalueFilter::FlipNegative);
    g += T * gf;
    H += T * Hf * T.Transpose();
    return dij;
}

/**
 * @brief Accumulate half-edge vertex to closest-point contact derivatives into gradient and
 * hessian.
 *
 * @param TMatrixXI Type for vertex position
 * @param TMatrixXJ Type for edge vertex position
 * @param TMatrixXTI Type for vertex position at previous time step
 * @param TMatrixXTJ Type for edge vertex position at previous time step
 * @param TMatrixUV Type for barycentric coordinates along edge
 * @param TMatrixXCP Type for closest point position
 * @param TMatrixXTCP Type for closest point position at previous time step
 * @param TMatrixG Type for gradient
 * @param TMatrixH Type for hessian
 * @param ScalarType Scalar type
 * @param xi `3 x 1` vertex position
 * @param xj `3 x 1` edge vertex position
 * @param xti `3 x 1` vertex position at previous time step
 * @param xtj `3 x 1` edge vertex position at previous time step
 * @param uv `2 x 1` barycentric coordinates along edge
 * @param xcp `3 x 1` closest point position
 * @param xtcp `3 x 1` closest point position at previous time step
 * @param r Contact radius
 * @param kc Collision penalty
 * @param kcp `kcp = tau*kc*(tau - r)^2`, where `tau = r/2`
 * @param b `b = kc/2*(r - tau)^2 + kcp*log(tau)`, where `tau = r/2`
 * @param mu Friction coefficient
 * @param epsvh Time-step scaled relative velocity threshold for static to dynamic friction
 * transition
 * @param h2inv Inverse squared time step
 * @param g `3 x 1` gradient
 * @param H `3 x 3` hessian
 */
template <
    mini::CMatrix TMatrixXI,
    mini::CMatrix TMatrixXJ,
    mini::CMatrix TMatrixXTI,
    mini::CMatrix TMatrixXTJ,
    mini::CMatrix TMatrixUV,
    mini::CMatrix TMatrixXCP,
    mini::CMatrix TMatrixXTCP,
    mini::CMatrix TMatrixG,
    mini::CMatrix TMatrixH,
    class ScalarType = typename TMatrixXI::ScalarType>
PBAT_HOST_DEVICE ScalarType AccumulateHalfEdgeVertexToClosestPointContactDerivatives(
    TMatrixXI const& xi,
    TMatrixXJ const& xj,
    TMatrixXTI const& xti,
    TMatrixXTJ const& xtj,
    TMatrixUV const& uv,
    int ilocal,
    TMatrixXCP const& xcp,
    TMatrixXTCP const& xtcp,
    ScalarType r,
    ScalarType kc,
    ScalarType kcp,
    ScalarType b,
    ScalarType mu,
    ScalarType epsvh,
    ScalarType h2inv,
    TMatrixG& g,
    TMatrixH& H)
{
    using namespace mini;
    // Normal contact
    mini::SVector<ScalarType, 3> x  = uv(0) * xi + uv(1) * xj;
    mini::SVector<ScalarType, 3> xt = uv(0) * xti + uv(1) * xtj;
    ScalarType d                    = Norm(x - xcp);
    mini::SVector<ScalarType, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    mini::SVector<ScalarType, 3> dBdx =
        contact::potentials::GradientSegmentWrtLinearlyInterpolatedClosestPoints(
            uv,
            x,
            xcp,
            d,
            dBdd(1),
            ilocal);
    mini::SMatrix<ScalarType, 3, 3> d2Bdx2 =
        contact::potentials::HessianBlockWrtLinearlyInterpolatedClosestPoints(
            uv,
            x,
            xcp,
            d,
            dBdd(1),
            dBdd(2),
            ilocal,
            ilocal);
    g += h2inv * dBdx;
    H += h2inv * d2Bdx2;
    // Frictional contact
    contact::potentials::LaggedFriction friction{};
    SMatrix<ScalarType, 3, 2> const T = contact::PointPointTangentialBasis(x, xcp);
    SVector<ScalarType, 2> const uk   = T.Transpose() * ((x - xt) - (xcp - xtcp));
    SVector<ScalarType, 2> gf;
    SMatrix<ScalarType, 2, 2> Hf;
    ScalarType lambda = -dBdd(1) * h2inv;
    friction.GradAndHessian(uk, mu, lambda, epsvh, gf, Hf);
    Hf = math::linalg::FilterEigenvalues(Hf, math::linalg::EEigenvalueFilter::FlipNegative);
    g += uv(0) * (T * gf);
    H += (uv(0) * uv(0)) * (T * Hf * T.Transpose());
    return d;
}

/**
 * @brief Accumulate triangle vertex to closest-point contact derivatives into gradient and hessian.
 * @tparam TMatrixXA Type for first triangle vertex position
 * @tparam TMatrixXB Type for second triangle vertex position
 * @tparam TMatrixXC Type for third triangle vertex position
 * @tparam TMatrixXTA Type for first triangle vertex position at previous time step
 * @tparam TMatrixXTB Type for second triangle vertex position at previous time step
 * @tparam TMatrixXTC Type for third triangle vertex position at previous time step
 * @tparam TMatrixUVW Type for barycentric coordinates within triangle
 * @tparam TMatrixXCP Type for closest point position
 * @tparam TMatrixXTCP Type for closest point position at previous time step
 * @tparam TMatrixG Type for gradient
 * @tparam TMatrixH Type for hessian
 * @tparam ScalarType Scalar type
 * @param xa `3 x 1` first triangle vertex position
 * @param xb `3 x 1` second triangle vertex position
 * @param xc `3 x 1` third triangle vertex position
 * @param xta `3 x 1` first triangle vertex position at previous time step
 * @param xtb `3 x 1` second triangle vertex position at previous time step
 * @param xtc `3 x 1` third triangle vertex position at previous time step
 * @param uvw `3 x 1` barycentric coordinates within triangle
 * @param ilocal Local index of closest point on triangle (0, 1, or 2)
 * @param xcp `3 x 1` closest point position
 * @param xtcp `3 x 1` closest point position at previous time step
 * @param r Contact radius
 * @param kc Collision penalty
 * @param kcp `kcp = tau*kc*(tau - r)^2`, where `tau = r/2`
 * @param b `b = kc/2*(r - tau)^2 + kcp*log(tau)`, where `tau = r/2`
 * @param mu Friction coefficient
 * @param epsvh Time-step scaled relative velocity threshold for static to dynamic friction
 * transition
 * @param h2inv Inverse squared time step
 * @param g `3 x 1` gradient
 * @param H `3 x 3` hessian
 */
template <
    mini::CMatrix TMatrixXA,
    mini::CMatrix TMatrixXB,
    mini::CMatrix TMatrixXC,
    mini::CMatrix TMatrixXTA,
    mini::CMatrix TMatrixXTB,
    mini::CMatrix TMatrixXTC,
    mini::CMatrix TMatrixUVW,
    mini::CMatrix TMatrixXCP,
    mini::CMatrix TMatrixXTCP,
    mini::CMatrix TMatrixG,
    mini::CMatrix TMatrixH,
    class ScalarType = typename TMatrixXA::ScalarType>
PBAT_HOST_DEVICE ScalarType AccumulateTriangleVertexToClosestPointContactDerivatives(
    TMatrixXA const& xa,
    TMatrixXB const& xb,
    TMatrixXC const& xc,
    TMatrixXTA const& xta,
    TMatrixXTB const& xtb,
    TMatrixXTC const& xtc,
    TMatrixUVW const& uvw,
    int ilocal,
    TMatrixXCP const& xcp,
    TMatrixXTCP const& xtcp,
    ScalarType r,
    ScalarType kc,
    ScalarType kcp,
    ScalarType b,
    ScalarType mu,
    ScalarType epsvh,
    ScalarType h2inv,
    TMatrixG& g,
    TMatrixH& H)
{
    using namespace mini;
    // Normal contact
    SVector<ScalarType, 3> x  = uvw(0) * xa + uvw(1) * xb + uvw(2) * xc;
    SVector<ScalarType, 3> xt = uvw(0) * xta + uvw(1) * xtb + uvw(2) * xtc;
    ScalarType d              = Norm(x - xcp);
    SVector<ScalarType, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    SVector<ScalarType, 3> dBdx =
        contact::potentials::GradientSegmentWrtLinearlyInterpolatedClosestPoints(
            uvw,
            x,
            xcp,
            d,
            dBdd(1),
            ilocal);
    SMatrix<ScalarType, 3, 3> d2Bdx2 =
        contact::potentials::HessianBlockWrtLinearlyInterpolatedClosestPoints(
            uvw,
            x,
            xcp,
            d,
            dBdd(1),
            dBdd(2),
            ilocal,
            ilocal);
    g += h2inv * dBdx;
    H += h2inv * d2Bdx2;
    // Frictional contact
    contact::potentials::LaggedFriction friction{};
    SMatrix<ScalarType, 3, 2> const T = contact::PointPointTangentialBasis(x, xcp);
    SVector<ScalarType, 2> const uk   = T.Transpose() * ((x - xt) - (xcp - xtcp));
    SVector<ScalarType, 2> gf;
    SMatrix<ScalarType, 2, 2> Hf;
    ScalarType lambda = -dBdd(1) * h2inv;
    friction.GradAndHessian(uk, mu, lambda, epsvh, gf, Hf);
    Hf = math::linalg::FilterEigenvalues(Hf, math::linalg::EEigenvalueFilter::FlipNegative);
    g += uvw(0) * (T * gf);
    H += (uvw(0) * uvw(0)) * (T * Hf * T.Transpose());
    return d;
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

/**
 * @brief Accumulate augmented Lagrangian contact gradient and Hessian for a single stencil node.
 *
 * Given a linearized contact constraint with gradient `gradc` (kDofs x 1), tangent basis `T` (3x2),
 * friction weights, and augmented Lagrangian parameters, accumulates the per-vertex (3x1 gradient,
 * 3x3 Hessian) contribution for stencil node `ki`.
 *
 * Normal:   gi += dL * gradc_i,   Hi += kn * gradc_i * gradc_i^T
 * Friction: gi += Wki * T * df,   Hi += kf * Wki^2 * T * T^T
 *
 * @tparam kDims Spatial dimensions (typically 3)
 * @param gradc Full stencil constraint gradient (kDofs x 1)
 * @param ki Local stencil index of the vertex
 * @param dL Normal contact multiplier: kn*cs - lambda
 * @param kn Normal penalty parameter: gamma*kc
 * @param T Tangent basis at contact point (kDims x 2)
 * @param Wki Friction weight for stencil node ki
 * @param kf Friction penalty parameter: gammaf*kc
 * @param df Friction dual residual: kf*cf - lambdaf (2 x 1)
 * @param gi Per-vertex gradient accumulator (kDims x 1)
 * @param Hi Per-vertex Hessian accumulator (kDims x kDims)
 */
template <
    int kDims,
    mini::CMatrix TGradC,
    mini::CMatrix TMatrixT,
    mini::CMatrix TDf,
    mini::CMatrix TMatrixG,
    mini::CMatrix TMatrixH,
    class ScalarType = typename TGradC::ScalarType>
PBAT_HOST_DEVICE void AccumulateAugmentedLagrangianContactNodeDerivatives(
    TGradC const& gradc,
    int ki,
    ScalarType dL,
    ScalarType kn,
    TMatrixT const& T,
    ScalarType Wki,
    ScalarType kf,
    TDf const& df,
    ScalarType decay,
    TMatrixG& gi,
    TMatrixH& Hi)
{
    auto gradci = gradc.template Slice<kDims, 1>(ki * kDims, 0);
    // Normal contact
    gi += decay * dL * gradci;
    Hi += decay * kn * (gradci * gradci.Transpose());
    // Friction contact
    gi += decay * Wki * (T * df);
    Hi += decay * (kf * Wki * Wki) * (T * T.Transpose());
}

} // namespace pbat::sim::algorithm::vbd::kernels

#endif // PBAT_SIM_ALGORITHM_VBD_KERNELS_H
