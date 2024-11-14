#ifndef PBAT_SIM_XPBD_KERNELS_H
#define PBAT_SIM_XPBD_KERNELS_H

#include "pbat/common/ConstexprFor.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/IntersectionQueries.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <algorithm>

namespace pbat {
namespace sim {
namespace xpbd {
namespace kernels {

namespace mini = pbat::math::linalg::mini;

template <
    mini::CMatrix TMatrixXT,
    mini::CMatrix TMatrixVT,
    mini::CMatrix TMatrixA,
    class ScalarType = typename TMatrixXT::ScalarType>
PBAT_HOST_DEVICE mini::SVector<ScalarType, TMatrixXT::kRows> InitialPosition(
    TMatrixXT const& xt,
    TMatrixVT const& vt,
    TMatrixA const& aext,
    ScalarType dt,
    ScalarType dt2)
{
    return xt + dt * vt + dt2 * aext;
}

template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    class ScalarType = typename TMatrixA::ScalarType>
PBAT_HOST_DEVICE mini::SVector<ScalarType, 2> Solve2x2(TMatrixA const& A, TMatrixB const& b)
{
    mini::SVector<ScalarType, 2> x;
    // NOTE: Add partial pivoting if conditioning problems
    ScalarType const A2 = (A(1, 1) - (A(1, 0) * A(0, 1) / A(0, 0)));
    ScalarType const b2 = b(1) - (A(1, 0) * b(0) / A(0, 0));
    x(1)                = b2 / A2;
    x(0)                = (b(0) - A(0, 1) * x(1)) / A(0, 0);
    return x;
}

template <
    mini::CMatrix TMatrixMinv,
    mini::CMatrix TMatrixDmInv,
    mini::CMatrix TMatrixAlphaT,
    mini::CMatrix TMatrixGamma,
    mini::CMatrix TMatrixXTC,
    mini::CMatrix TMatrixL,
    mini::CMatrix TMatrixXC,
    class ScalarType = typename TMatrixMinv::ScalarType>
PBAT_HOST_DEVICE void ProjectBlockNeoHookean(
    TMatrixMinv const& minvc,
    TMatrixDmInv const& DmInv,
    ScalarType gammaSNHc,
    TMatrixAlphaT atildec,
    TMatrixGamma gammac,
    TMatrixXTC const& xtc,
    TMatrixL& lambdac,
    TMatrixXC& xc)
{
    using namespace mini;
    // Compute deviatoric+hydrostatic elasticity
#if defined(CUDART_VERSION)
    #pragma nv_diag_suppress 174
#endif
    SMatrix<ScalarType, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInv;
    ScalarType CD               = Norm(F);
    SMatrix<ScalarType, 3, 4> gradCD{};
    gradCD.Slice<3, 3>(0, 1) = (F * DmInv.Transpose()) / (CD /*+ 1e-8*/);
    gradCD.Col(0)            = -(gradCD.Col(1) + gradCD.Col(2) + gradCD.Col(3));
    ScalarType CH            = Determinant(F) - gammaSNHc;
    SMatrix<ScalarType, 3, 3> PH{};
    PH.Col(0) = Cross(F.Col(1), F.Col(2));
    PH.Col(1) = Cross(F.Col(2), F.Col(0));
    PH.Col(2) = Cross(F.Col(0), F.Col(1));
    SMatrix<ScalarType, 3, 4> gradCH{};
    gradCH.Slice<3, 3>(0, 1) = PH * DmInv.Transpose();
    gradCH.Col(0)            = -(gradCH.Col(1) + gradCH.Col(2) + gradCH.Col(3));
#if defined(CUDART_VERSION)
    #pragma nv_diag_default 174
#endif
    // Construct 2x2 constraint block system
    SVector<ScalarType, 2> b{
        -(CD + atildec(0) * lambdac(0) + gammac(0) * Dot(gradCD, xc - xtc)),
        -(CH + atildec(1) * lambdac(1) + gammac(1) * Dot(gradCH, xc - xtc))};
    SVector<ScalarType, 2> D = Ones<ScalarType, 2, 1>() + gammac;
    SMatrix<ScalarType, 2, 2> A{};
    A(0, 0) =
        D(0) * (minvc(0) * SquaredNorm(gradCD.Col(0)) + minvc(1) * SquaredNorm(gradCD.Col(1)) +
                minvc(2) * SquaredNorm(gradCD.Col(2)) + minvc(3) * SquaredNorm(gradCD.Col(3))) +
        atildec(0);
    A(1, 1) =
        D(1) * (minvc(0) * SquaredNorm(gradCH.Col(0)) + minvc(1) * SquaredNorm(gradCH.Col(1)) +
                minvc(2) * SquaredNorm(gradCH.Col(2)) + minvc(3) * SquaredNorm(gradCH.Col(3))) +
        atildec(1);
    A(0, 1) =
        (minvc(0) * Dot(gradCD.Col(0), gradCH.Col(0)) +
         minvc(1) * Dot(gradCD.Col(1), gradCH.Col(1)) +
         minvc(2) * Dot(gradCD.Col(2), gradCH.Col(2)) +
         minvc(3) * Dot(gradCD.Col(3), gradCH.Col(3)));
    A(1, 0) = A(0, 1);
    A(0, 1) *= D(0);
    A(1, 0) *= D(1);
    // Project block constraint
    SVector<ScalarType, 2> dlambda = Solve2x2(A, b);
    lambdac += dlambda;
    pbat::common::ForRange<0, 4>([&]<auto i>() {
        xc.Col(i) += minvc(i) * (dlambda(0) * gradCD.Col(i) + dlambda(1) * gradCH.Col(i));
    });
}

template <
    mini::CMatrix TMatrixXVT,
    mini::CMatrix TMatrixXFT,
    mini::CMatrix TMatrixXF,
    mini::CMatrix TMatrixXV,
    class ScalarType = typename TMatrixXV::ScalarType>
PBAT_HOST_DEVICE bool ProjectVertexTriangle(
    ScalarType minvv,
    TMatrixXVT const& xvt,
    TMatrixXFT const& xft,
    TMatrixXF const& xf,
    ScalarType muC,
    ScalarType muS,
    ScalarType muD,
    ScalarType atildec,
    ScalarType gammac,
    ScalarType& lambdac,
    TMatrixXV& xv)
{
    using namespace mini;
    // Numerically zero inverse mass makes the Schur complement ill-conditioned/singular
    if (minvv < ScalarType(1e-10))
        return false;

    // Compute triangle normal
    SMatrix<ScalarType, 3, 1> T1     = xf.Col(1) - xf.Col(0);
    SMatrix<ScalarType, 3, 1> T2     = xf.Col(2) - xf.Col(0);
    SMatrix<ScalarType, 3, 1> n      = Cross(T1, T2);
    ScalarType const doublearea      = Norm(n);
    bool const bIsTriangleDegenerate = doublearea <= ScalarType(1e-8);
    if (bIsTriangleDegenerate)
        return false;

    n /= doublearea;
    using namespace pbat::geometry;
    SMatrix<ScalarType, 3, 1> xc = ClosestPointQueries::PointOnPlane(xv, xf.Col(0), n);
    // Check if xv projects to the triangle's interior by checking its barycentric coordinates
    SMatrix<ScalarType, 3, 1> b =
        IntersectionQueries::TriangleBarycentricCoordinates(xc - xf.Col(0), T1, T2);
    // If xv doesn't project inside triangle, then we don't generate a contact response
    // clang-format off
    bool const bIsVertexInsideTriangle = 
        (b(0) >= ScalarType(0)) and (b(0) <= ScalarType(1)) and
        (b(1) >= ScalarType(0)) and (b(1) <= ScalarType(1)) and
        (b(2) >= ScalarType(0)) and (b(2) <= ScalarType(1));
    // clang-format on
    if (not bIsVertexInsideTriangle)
        return false;

    // Project xv onto triangle's plane into xc
    ScalarType const C = muC * Dot(n, xv - xf.Col(0));
    // If xv is positively oriented w.r.t. triangles xf, there is no penetration
    if (C > ScalarType(0))
        return false;
    // Prevent super violent projection for stability, i.e. if the collision constraint
    // violation is already too large, we give up.
    // if (C < -1e-2)
    //     return false;

    // Project constraints:
    // We assume that the triangle is static (although it is not), so that the gradient is n for
    // the vertex.

    // Collision constraint
    ScalarType const D = ScalarType(1) + gammac;
    ScalarType dlambda =
        -(C + atildec * lambdac + gammac * Dot(n, xv - xvt)) / (D * minvv + atildec);
    SMatrix<ScalarType, 3, 1> dx = dlambda * minvv * n;
    xv += dx;
    lambdac += dlambda;

    // Friction constraint (see https://dl.acm.org/doi/10.1145/2601097.2601152)
    ScalarType const d   = Norm(dx);
    dx                   = (xv - xvt) - (xf * b - xft * b);
    dx                   = dx - n * n.Transpose() * dx;
    ScalarType const dxd = Norm(dx);
    if (dxd > muS * d)
    {
        using namespace std;
        dx *= min(muD * d / dxd, ScalarType(1));
    }

    xv += dx;
    return true;
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

} // namespace kernels
} // namespace xpbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_XPBD_KERNELS_H