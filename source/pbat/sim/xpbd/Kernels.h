#ifndef PBAT_SIM_XPBD_KERNELS_H
#define PBAT_SIM_XPBD_KERNELS_H

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
    mini::CMatrix TMatrixGC,
    mini::CMatrix TMatrixMinvC,
    mini::CMatrix TMatrixXC,
    mini::CMatrix TMatrixXTC,
    class ScalarType = typename TMatrixGC::ScalarType>
PBAT_DEVICE void ProjectTetrahedron(
    ScalarType C,
    TMatrixGC const& gradC,
    TMatrixMinvC const& minvc,
    ScalarType atilde,
    ScalarType gammac,
    TMatrixXTC const& xtc,
    ScalarType& lambdac,
    TMatrixXC& xc)
{
    using namespace mini;
    ScalarType const D = ScalarType(1) + gammac;
    ScalarType dlambda =
        -(C + atilde * lambdac + gammac * Dot(gradC, xc - xtc)) /
        (D * minvc(0) * SquaredNorm(gradC.Col(0)) + D * minvc(1) * SquaredNorm(gradC.Col(1)) +
         D * minvc(2) * SquaredNorm(gradC.Col(2)) + D * minvc(3) * SquaredNorm(gradC.Col(3)) +
         atilde);
    lambdac += dlambda;
    xc.Col(0) += (minvc(0) * dlambda) * gradC.Col(0);
    xc.Col(1) += (minvc(1) * dlambda) * gradC.Col(1);
    xc.Col(2) += (minvc(2) * dlambda) * gradC.Col(2);
    xc.Col(3) += (minvc(3) * dlambda) * gradC.Col(3);
}

template <
    class IndexType,
    mini::CMatrix TMatrixMinv,
    mini::CMatrix TMatrixDmInv,
    mini::CMatrix TMatrixXTC,
    mini::CMatrix TMatrixXC,
    class ScalarType = typename TMatrixMinv::ScalarType>
PBAT_DEVICE void ProjectHydrostatic(
    IndexType c,
    TMatrixMinv const& minvc,
    TMatrixDmInv const& DmInv,
    ScalarType atilde,
    ScalarType gammaSNHc,
    ScalarType gammac,
    TMatrixXTC const& xtc,
    ScalarType& lambdac,
    TMatrixXC& xc)
{
    using namespace mini;
#if defined(CUDART_VERSION)
    #pragma nv_diag_suppress 174
#endif
    SMatrix<ScalarType, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInv;
    ScalarType C                = Determinant(F) - gammaSNHc;
    SMatrix<ScalarType, 3, 3> P{};
    P.Col(0) = Cross(F.Col(1), F.Col(2));
    P.Col(1) = Cross(F.Col(2), F.Col(0));
    P.Col(2) = Cross(F.Col(0), F.Col(1));
    SMatrix<ScalarType, 3, 4> gradC{};
    gradC.Slice<3, 3>(0, 1) = P * DmInv.Transpose();
    gradC.Col(0)            = -(gradC.Col(1) + gradC.Col(2) + gradC.Col(3));
    ProjectTetrahedron(C, gradC, minvc, atilde, gammac, xtc, lambdac, xc);
#if defined(CUDART_VERSION)
    #pragma nv_diag_default 174
#endif
}

template <
    class IndexType,
    mini::CMatrix TMatrixMinv,
    mini::CMatrix TMatrixDmInv,
    mini::CMatrix TMatrixXTC,
    mini::CMatrix TMatrixXC,
    class ScalarType = typename TMatrixMinv::ScalarType>
PBAT_DEVICE void ProjectDeviatoric(
    IndexType c,
    TMatrixMinv const& minvc,
    TMatrixDmInv const& DmInv,
    ScalarType atilde,
    ScalarType gammac,
    TMatrixXTC const& xtc,
    ScalarType& lambdac,
    TMatrixXC& xc)
{
    using namespace mini;
#if defined(CUDART_VERSION)
    #pragma nv_diag_suppress 174
#endif
    SMatrix<ScalarType, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInv;
    ScalarType C                = Norm(F);
    SMatrix<ScalarType, 3, 4> gradC{};
    gradC.Slice<3, 3>(0, 1) = (F * DmInv.Transpose()) / (C /*+ 1e-8*/);
    gradC.Col(0)            = -(gradC.Col(1) + gradC.Col(2) + gradC.Col(3));
    ProjectTetrahedron(C, gradC, minvc, atilde, gammac, xtc, lambdac, xc);
#if defined(CUDART_VERSION)
    #pragma nv_diag_default 174
#endif
}

template <
    mini::CMatrix TMatrixMinvF,
    mini::CMatrix TMatrixXVT,
    mini::CMatrix TMatrixXFT,
    mini::CMatrix TMatrixXF,
    mini::CMatrix TMatrixXV,
    class ScalarType = typename TMatrixXV::ScalarType>
PBAT_DEVICE bool ProjectVertexTriangle(
    ScalarType minvv,
    TMatrixMinvF const& minvf,
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