#ifndef PBAT_GPU_XPBD_XPBD_IMPL_KERNELS_CUH
#define PBAT_GPU_XPBD_XPBD_IMPL_KERNELS_CUH

#include "XpbdImpl.cuh"
#include "pbat/HostDevice.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/IntersectionQueries.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/SynchronizedList.cuh"
#include "pbat/math/linalg/mini/Mini.h"

#include <array>

namespace pbat {
namespace gpu {
namespace xpbd {
namespace XpbdImplKernels {

namespace mini = pbat::math::linalg::mini;

struct FInitializeNeoHookeanConstraint
{
    PBAT_DEVICE void operator()(GpuIndex c)
    {
        using namespace mini;
        // Load vertex positions of element c
        SMatrix<GpuIndex, 4, 1> v   = FromBuffers<4, 1>(T, c);
        SMatrix<GpuScalar, 3, 4> xc = FromBuffers(x, v.Transpose());
        // Compute shape matrix and its inverse
        SMatrix<GpuScalar, 3, 3> Ds         = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0)));
        SMatrixView<GpuScalar, 3, 3> DmInvC = FromFlatBuffer<3, 3>(DmInv, c);
        DmInvC                              = Inverse(Ds);
        // Compute constraint compliance
        GpuScalar const tetVolume           = Determinant(Ds) / GpuScalar{6};
        SMatrixView<GpuScalar, 2, 1> alphac = FromFlatBuffer<2, 1>(alpha, c);
        SMatrixView<GpuScalar, 2, 1> lamec  = FromFlatBuffer<2, 1>(lame, c);
        alphac                              = GpuScalar{1} / (lamec * tetVolume);
        // Compute rest stability
        gamma[c] = GpuScalar{1.} + lamec(0) / lamec(1);
    }
    std::array<GpuScalar*, 3> x;
    std::array<GpuIndex*, 4> T;
    GpuScalar* lame;
    GpuScalar* DmInv;
    GpuScalar* alpha;
    GpuScalar* gamma;
};

struct FInitializeSolution
{
    PBAT_DEVICE void operator()(GpuIndex i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            x[d][i] = xt[d][i] + dt * v[d][i] + dt2 * minv[i] * f[d][i];
        }
    }

    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> x;
    std::array<GpuScalar*, 3> v;
    std::array<GpuScalar*, 3> f;
    GpuScalar* minv;
    GpuScalar dt;
    GpuScalar dt2;
};

struct FStableNeoHookeanConstraint
{
    PBAT_DEVICE void Project(
        GpuScalar C,
        mini::SMatrix<GpuScalar, 3, 4> const& gradC,
        mini::SMatrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar& lambdac,
        mini::SMatrix<GpuScalar, 3, 4>& xc)
    {
        using namespace mini;
        GpuScalar dlambda =
            -(C + atilde * lambdac) /
            (minvc(0) * SquaredNorm(gradC.Col(0)) + minvc(1) * SquaredNorm(gradC.Col(1)) +
             minvc(2) * SquaredNorm(gradC.Col(2)) + minvc(3) * SquaredNorm(gradC.Col(3)) + atilde);
        lambdac += dlambda;
        xc.Col(0) += (minvc(0) * dlambda) * gradC.Col(0);
        xc.Col(1) += (minvc(1) * dlambda) * gradC.Col(1);
        xc.Col(2) += (minvc(2) * dlambda) * gradC.Col(2);
        xc.Col(3) += (minvc(3) * dlambda) * gradC.Col(3);
    }

    PBAT_DEVICE void ProjectHydrostatic(
        GpuIndex c,
        mini::SMatrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar gammac,
        GpuScalar& lambdac,
        mini::SMatrix<GpuScalar, 3, 4>& xc)
    {
        using namespace mini;
        SMatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        SMatrix<GpuScalar, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInvC;
        GpuScalar C                = Determinant(F) - gammac;
        SMatrix<GpuScalar, 3, 3> P{};
        P.Col(0) = Cross(F.Col(1), F.Col(2));
        P.Col(1) = Cross(F.Col(2), F.Col(0));
        P.Col(2) = Cross(F.Col(0), F.Col(1));
        SMatrix<GpuScalar, 3, 4> gradC{};
        gradC.Slice<3, 3>(0, 1) = P * DmInvC.Transpose();
        gradC.Col(0)            = -(gradC.Col(1) + gradC.Col(2) + gradC.Col(3));
        Project(C, gradC, minvc, atilde, lambdac, xc);
    }

    PBAT_DEVICE void ProjectDeviatoric(
        GpuIndex c,
        mini::SMatrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar& lambdac,
        mini::SMatrix<GpuScalar, 3, 4>& xc)
    {
        using namespace mini;
        SMatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        SMatrix<GpuScalar, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInvC;
        GpuScalar C                = Norm(F);
        SMatrix<GpuScalar, 3, 4> gradC{};
        gradC.Slice<3, 3>(0, 1) = (F * DmInvC.Transpose()) / (C /*+ 1e-8*/);
        gradC.Col(0)            = -(gradC.Col(1) + gradC.Col(2) + gradC.Col(3));
        Project(C, gradC, minvc, atilde, lambdac, xc);
    }

    PBAT_DEVICE void operator()(GpuIndex c)
    {
        using namespace mini;

        // 1. Load constraint data in local memory
        SMatrix<GpuIndex, 4, 1> v        = FromBuffers<4, 1>(T, c);
        SMatrix<GpuScalar, 3, 4> xc      = FromBuffers(x, v.Transpose());
        SMatrix<GpuScalar, 4, 1> minvc   = FromFlatBuffer(minv, v);
        SMatrix<GpuScalar, 2, 1> lambdac = FromFlatBuffer<2, 1>(lambda, c);
        SMatrix<GpuScalar, 2, 1> atilde  = FromFlatBuffer<2, 1>(alpha, c);
        atilde /= dt2;

        // 2. Project elastic constraints
        ProjectDeviatoric(c, minvc, atilde(0), lambdac(0), xc);
        ProjectHydrostatic(c, minvc, atilde(1), gamma[c], lambdac(1), xc);

        // 3. Update global "Lagrange" multipliers and positions
        ToFlatBuffer(lambdac, lambda, c);
        ToBuffers(xc, v.Transpose(), x);
    }

    std::array<GpuScalar*, 3> x;
    GpuScalar* lambda;

    std::array<GpuIndex*, 4> T;
    GpuScalar* minv;
    GpuScalar* alpha;
    GpuScalar* DmInv;
    GpuScalar* gamma;
    GpuScalar dt2;
};

struct FVertexTriangleContactConstraint
{
    using ContactPairType = typename XpbdImpl::ContactPairType;

    PBAT_DEVICE bool ProjectVertexTriangle(
        GpuScalar minvv,
        mini::SMatrix<GpuScalar, 3, 1> const& minvf,
        mini::SMatrix<GpuScalar, 3, 1> const& xvt,
        mini::SMatrix<GpuScalar, 3, 3> const& xft,
        mini::SMatrix<GpuScalar, 3, 3> const& xf,
        GpuScalar atildec,
        GpuScalar& lambdac,
        mini::SMatrix<GpuScalar, 3, 1>& xv)
    {
        using namespace mini;
        // Numerically zero inverse mass makes the Schur complement ill-conditioned/singular
        if (minvv < GpuScalar{1e-10})
            return false;
        // Compute triangle normal
        SMatrix<GpuScalar, 3, 1> T1      = xf.Col(1) - xf.Col(0);
        SMatrix<GpuScalar, 3, 1> T2      = xf.Col(2) - xf.Col(0);
        SMatrix<GpuScalar, 3, 1> n       = Cross(T1, T2);
        GpuScalar const doublearea       = Norm(n);
        bool const bIsTriangleDegenerate = doublearea <= GpuScalar{1e-8f};
        if (bIsTriangleDegenerate)
            return false;

        n /= doublearea;
        using namespace pbat::geometry;
        SMatrix<GpuScalar, 3, 1> xc = ClosestPointQueries::PointOnPlane(xv, xf.Col(0), n);
        // Check if xv projects to the triangle's interior by checking its barycentric coordinates
        SMatrix<GpuScalar, 3, 1> b =
            IntersectionQueries::TriangleBarycentricCoordinates(xc - xf.Col(0), T1, T2);
        // If xv doesn't project inside triangle, then we don't generate a contact response
        // clang-format off
        bool const bIsVertexInsideTriangle = 
            (b(0) >= GpuScalar{0}) and (b(0) <= GpuScalar{1}) and
            (b(1) >= GpuScalar{0}) and (b(1) <= GpuScalar{1}) and
            (b(2) >= GpuScalar{0}) and (b(2) <= GpuScalar{1});
        // clang-format on
        if (not bIsVertexInsideTriangle)
            return false;

        // Project xv onto triangle's plane into xc
        GpuScalar const C = Dot(n, xv - xf.Col(0));
        // If xv is positively oriented w.r.t. triangles xf, there is no penetration
        if (C > GpuScalar{0})
            return false;
        // Prevent super violent projection for stability, i.e. if the collision constraint
        // violation is already too large, we give up.
        // if (C < -kMaxPenetration)
        //     return false;

        // Project constraints:
        // We assume that the triangle is static (although it is not), so that the gradient is n for
        // the vertex.

        // Collision constraint
        GpuScalar dlambda           = -(C + atildec * lambdac) / (minvv + atildec);
        SMatrix<GpuScalar, 3, 1> dx = dlambda * minvv * n;
        xv += dx;
        lambdac += dlambda;

        // Friction constraint (see https://dl.acm.org/doi/10.1145/2601097.2601152)
        GpuScalar const d   = Norm(dx);
        dx                  = (xv - xvt) - (xf * b - xft * b);
        dx                  = dx - n * n.Transpose() * dx;
        GpuScalar const dxd = Norm(dx);
        if (dxd > muS * d)
            dx *= min(muK * d / dxd, 1.);

        xv += dx;
        return true;
    }

    PBAT_DEVICE void operator()(GpuIndex c)
    {
        using namespace mini;
        GpuIndex vv = V[0][cpairs[c].first]; ///< Colliding vertex
        SMatrix<GpuIndex, 3, 1> vf{
            F[0][cpairs[c].second],
            F[1][cpairs[c].second],
            F[2][cpairs[c].second]}; ///< Colliding triangle

        SMatrix<GpuScalar, 3, 1> xvt = FromBuffers<3, 1>(xt, vv);
        SMatrix<GpuScalar, 3, 1> xv  = FromBuffers<3, 1>(x, vv);
        SMatrix<GpuScalar, 3, 3> xft = FromBuffers(xt, vf.Transpose());
        SMatrix<GpuScalar, 3, 3> xf  = FromBuffers(x, vf.Transpose());

        GpuScalar const minvv          = minv[vv];
        SMatrix<GpuScalar, 3, 1> minvf = FromFlatBuffer(minv, vf);
        GpuScalar atildec              = alpha[c] / dt2;
        GpuScalar lambdac              = lambda[c];

        // 2. Project collision constraint
        if (not ProjectVertexTriangle(minvv, minvf, xvt, xft, xf, atildec, lambdac, xv))
            return;

        // 3. Update global positions
        // lambda[c] = lambdac;
        ToBuffers(xv, x, vv);
    }

    std::array<GpuScalar*, 3> x;                                  ///< Current positions
    GpuScalar* lambda;                                            ///< Lagrange multipliers
    common::DeviceSynchronizedList<ContactPairType> const cpairs; ///< Vertex-triangle contacts
    std::array<GpuIndex*, 4> V;                                   ///< Vertex indices
    std::array<GpuIndex*, 4> F;                                   ///< Triangle indices
    GpuScalar* minv;                                              ///< Inverse mass
    GpuScalar* alpha;                                             ///< Compliance
    std::array<GpuScalar*, 3> xt;                                 ///< Positions at time t
    GpuScalar dt2;                                                ///< Squared time step
    GpuScalar const muS; ///< Static Coulomb friction coefficient
    GpuScalar const muK; ///< Dynamic Coulomb friction coefficient
};

struct FUpdateSolution
{
    PBAT_DEVICE void operator()(GpuIndex i)
    {
        for (auto d = 0; d < 3; ++d)
        {
            v[d][i]  = (x[d][i] - xt[d][i]) / dt;
            xt[d][i] = x[d][i];
        }
    }

    std::array<GpuScalar*, 3> xt;
    std::array<GpuScalar*, 3> x;
    std::array<GpuScalar*, 3> v;
    GpuScalar dt;
};

} // namespace XpbdImplKernels
} // namespace xpbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_XPBD_XPBD_IMPL_KERNELS_CUH