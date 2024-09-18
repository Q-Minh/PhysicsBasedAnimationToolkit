#ifndef PBAT_GPU_XPBD_XPBD_IMPL_KERNELS_CUH
#define PBAT_GPU_XPBD_XPBD_IMPL_KERNELS_CUH

#include "XpbdImpl.cuh"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/SynchronizedList.cuh"
#include "pbat/gpu/math/linalg/Matrix.cuh"

#include <array>

namespace pbat {
namespace gpu {
namespace xpbd {
namespace XpbdImplKernels {

struct FInitializeNeoHookeanConstraint
{
    __device__ void operator()(GpuIndex c)
    {
        using namespace pbat::gpu::math::linalg;
        // Load vertex positions of element c
        GpuIndex const v[4] = {T[0][c], T[1][c], T[2][c], T[3][c]};
        Matrix<GpuScalar, 3, 4> xc{};
        for (auto d = 0; d < 3; ++d)
            for (auto j = 0; j < 4; ++j)
                xc(d, j) = x[d][v[j]];
        // Compute shape matrix and its inverse
        Matrix<GpuScalar, 3, 3> Ds = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0)));
        MatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        DmInvC = Inverse(Ds);
        // Compute constraint compliance
        GpuScalar const tetVolume = Determinant(Ds) / GpuScalar{6.};
        MatrixView<GpuScalar, 2, 1> alphac{alpha + 2 * c};
        MatrixView<GpuScalar, 2, 1> lamec{lame + 2 * c};
        alphac(0) = GpuScalar{1.} / (lamec(0) * tetVolume);
        alphac(1) = GpuScalar{1.} / (lamec(1) * tetVolume);
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
    __device__ void operator()(GpuIndex i)
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
    __device__ void Project(
        GpuScalar C,
        math::linalg::Matrix<GpuScalar, 3, 4> const& gradC,
        math::linalg::Matrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar& lambdac,
        math::linalg::Matrix<GpuScalar, 3, 4>& xc)
    {
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

    __device__ void ProjectHydrostatic(
        GpuIndex c,
        math::linalg::Matrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar gammac,
        GpuScalar& lambdac,
        math::linalg::Matrix<GpuScalar, 3, 4>& xc)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        Matrix<GpuScalar, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInvC;
        GpuScalar C               = Determinant(F) - gammac;
        Matrix<GpuScalar, 3, 3> P{};
        P.Col(0) = Cross(F.Col(1), F.Col(2));
        P.Col(1) = Cross(F.Col(2), F.Col(0));
        P.Col(2) = Cross(F.Col(0), F.Col(1));
        Matrix<GpuScalar, 3, 4> gradC{};
        gradC.Slice<3, 3>(0, 1) = P * DmInvC.Transpose();
        gradC.Col(0)            = -(gradC.Col(1) + gradC.Col(2) + gradC.Col(3));
        Project(C, gradC, minvc, atilde, lambdac, xc);
    }

    __device__ void ProjectDeviatoric(
        GpuIndex c,
        math::linalg::Matrix<GpuScalar, 4, 1> const& minvc,
        GpuScalar atilde,
        GpuScalar& lambdac,
        math::linalg::Matrix<GpuScalar, 3, 4>& xc)
    {
        using namespace pbat::gpu::math::linalg;
        MatrixView<GpuScalar, 3, 3> DmInvC(DmInv + 9 * c);
        Matrix<GpuScalar, 3, 3> F = (xc.Slice<3, 3>(0, 1) - Repeat<1, 3>(xc.Col(0))) * DmInvC;
        GpuScalar C               = Norm(F);
        Matrix<GpuScalar, 3, 4> gradC{};
        gradC.Slice<3, 3>(0, 1) = (F * DmInvC.Transpose()) / (C /*+ 1e-8*/);
        gradC.Col(0)            = -(gradC.Col(1) + gradC.Col(2) + gradC.Col(3));
        Project(C, gradC, minvc, atilde, lambdac, xc);
    }

    __device__ void operator()(GpuIndex c)
    {
        using namespace pbat::gpu::math::linalg;

        // 1. Load constraint data in local memory
        GpuIndex const v[4] = {T[0][c], T[1][c], T[2][c], T[3][c]};
        Matrix<GpuScalar, 3, 4> xc{};
        for (auto d = 0; d < 3; ++d)
            for (auto j = 0; j < 4; ++j)
                xc(d, j) = x[d][v[j]];
        Matrix<GpuScalar, 4, 1> minvc{};
        for (auto j = 0; j < 4; ++j)
            minvc(j) = minv[v[j]];
        Matrix<GpuScalar, 2, 1> lambdac{};
        lambdac(0) = lambda[2 * c];
        lambdac(1) = lambda[2 * c + 1];
        Matrix<GpuScalar, 2, 1> atilde{};
        atilde(0) = alpha[2 * c] / dt2;
        atilde(1) = alpha[2 * c + 1] / dt2;

        // 2. Project elastic constraints
        ProjectDeviatoric(c, minvc, atilde(0), lambdac(0), xc);
        ProjectHydrostatic(c, minvc, atilde(1), gamma[c], lambdac(1), xc);

        // 3. Update global "Lagrange" multipliers and positions
        lambda[2 * c]     = lambdac(0);
        lambda[2 * c + 1] = lambdac(1);
        for (auto d = 0; d < 3; ++d)
            for (auto j = 0; j < 4; ++j)
                x[d][v[j]] = xc(d, j);
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

__device__ math::linalg::Matrix<GpuScalar, 3, 1>
PointOnPlane(auto const& X, auto const& P, auto const& n)
{
    using namespace pbat::gpu::math::linalg;
    GpuScalar const t          = (n.Transpose() * (X - P))(0, 0);
    Matrix<GpuScalar, 3, 1> Xp = X - t * n;
    return Xp;
}

__device__ math::linalg::Matrix<GpuScalar, 3, 1>
TriangleBarycentricCoordinates(auto const& AP, auto const& AB, auto const& AC)
{
    using namespace pbat::gpu::math::linalg;
    GpuScalar const d00   = Dot(AB, AB);
    GpuScalar const d01   = Dot(AB, AC);
    GpuScalar const d11   = Dot(AC, AC);
    GpuScalar const d20   = Dot(AP, AB);
    GpuScalar const d21   = Dot(AP, AC);
    GpuScalar const denom = d00 * d11 - d01 * d01;
    GpuScalar const v     = (d11 * d20 - d01 * d21) / denom;
    GpuScalar const w     = (d00 * d21 - d01 * d20) / denom;
    GpuScalar const u     = GpuScalar{1.} - v - w;
    Matrix<GpuScalar, 3, 1> uvw{};
    uvw(0, 0) = u;
    uvw(1, 0) = v;
    uvw(2, 0) = w;
    return uvw;
};

struct FVertexTriangleContactConstraint
{
    using ContactPairType = typename XpbdImpl::ContactPairType;

    __device__ bool ProjectVertexTriangle(
        GpuScalar minvv,
        math::linalg::Matrix<GpuScalar, 3, 1> const& minvf,
        math::linalg::Matrix<GpuScalar, 3, 1> const& xvt,
        math::linalg::Matrix<GpuScalar, 3, 3> const& xft,
        math::linalg::Matrix<GpuScalar, 3, 3> const& xf,
        GpuScalar atildec,
        GpuScalar& lambdac,
        math::linalg::Matrix<GpuScalar, 3, 1>& xv)
    {
        using namespace pbat::gpu::math::linalg;
        // Numerically zero inverse mass makes the Schur complement ill-conditioned/singular
        if (minvv < GpuScalar{1e-10})
            return false;
        // Compute triangle normal
        Matrix<GpuScalar, 3, 1> T1       = xf.Col(1) - xf.Col(0);
        Matrix<GpuScalar, 3, 1> T2       = xf.Col(2) - xf.Col(0);
        Matrix<GpuScalar, 3, 1> n        = Cross(T1, T2);
        GpuScalar const doublearea       = Norm(n);
        bool const bIsTriangleDegenerate = doublearea <= GpuScalar{1e-8f};
        if (bIsTriangleDegenerate)
            return false;

        n /= doublearea;
        Matrix<GpuScalar, 3, 1> xc = PointOnPlane(xv, xf.Col(0), n);
        // Check if xv projects to the triangle's interior by checking its barycentric coordinates
        Matrix<GpuScalar, 3, 1> b = TriangleBarycentricCoordinates(xc - xf.Col(0), T1, T2);
        // If xv doesn't project inside triangle, then we don't generate a contact response
        // clang-format off
        bool const bIsVertexInsideTriangle = 
            (b(0) >= GpuScalar{0.f}) and (b(0) <= GpuScalar{1.f}) and
            (b(1) >= GpuScalar{0.f}) and (b(1) <= GpuScalar{1.f}) and
            (b(2) >= GpuScalar{0.f}) and (b(2) <= GpuScalar{1.f});
        // clang-format on
        if (not bIsVertexInsideTriangle)
            return false;

        // Project xv onto triangle's plane into xc
        GpuScalar const C = Dot(n, xv - xf.Col(0));
        // If xv is positively oriented w.r.t. triangles xf, there is no penetration
        if (C > GpuScalar{0.})
            return false;
        // Prevent super violent projection for stability, i.e. if the collision constraint
        // violation is already too large, we give up.
        // if (C < -kMaxPenetration)
        //     return false;

        // Project constraints:
        // We assume that the triangle is static (although it is not), so that the gradient is n for
        // the vertex.

        // Collision constraint
        GpuScalar dlambda          = -(C + atildec * lambdac) / (minvv + atildec);
        Matrix<GpuScalar, 3, 1> dx = dlambda * minvv * n;
        xv += dx;
        lambdac += dlambda;

        // Friction constraint
        GpuScalar const d   = Norm(dx);
        dx                  = (xv - xvt) - (xf * b - xft * b);
        dx                  = dx - n * n.Transpose() * dx;
        GpuScalar const dxd = Norm(dx);
        if (dxd > muS * d)
            dx *= min(muK * d / dxd, 1.);

        xv += dx;
        return true;
    }

    __device__ void SetParticlePosition(
        GpuIndex v,
        math::linalg::Matrix<GpuScalar, 3, 1> const& xv,
        std::array<GpuScalar*, 3> const& xx)
    {
        for (auto d = 0; d < 3; ++d)
            xx[d][v] = xv(d, 0);
    }

    __device__ math::linalg::Matrix<GpuScalar, 3, 1>
    GetParticlePosition(GpuIndex v, std::array<GpuScalar*, 3> const& xx)
    {
        using namespace pbat::gpu::math::linalg;
        Matrix<GpuScalar, 3, 1> xv{};
        for (auto d = 0; d < 3; ++d)
            xv(d, 0) = xx[d][v];
        return xv;
    }

    __device__ void operator()(GpuIndex c)
    {
        using namespace pbat::gpu::math::linalg;
        GpuIndex vv    = V[0][cpairs[c].first]; ///< Colliding vertex
        GpuIndex vf[3] = {
            F[0][cpairs[c].second],
            F[1][cpairs[c].second],
            F[2][cpairs[c].second]}; ///< Colliding triangle

        Matrix<GpuScalar, 3, 1> xv = GetParticlePosition(vv, x);
        Matrix<GpuScalar, 3, 3> xf{};
        for (auto j = 0; j < 3; ++j)
            xf.Col(j) = GetParticlePosition(vf[j], x);

        Matrix<GpuScalar, 3, 1> xvt = GetParticlePosition(vv, xt);
        Matrix<GpuScalar, 3, 3> xft{};
        for (auto j = 0; j < 3; ++j)
            xft.Col(j) = GetParticlePosition(vf[j], xt);

        GpuScalar const minvv = minv[vv];
        Matrix<GpuScalar, 3, 1> minvf{};
        for (auto j = 0; j < 3; ++j)
            minvf(j, 0) = minv[vf[j]];
        GpuScalar atildec = alpha[c] / dt2;
        GpuScalar lambdac = lambda[c];

        // 2. Project collision constraint
        if (not ProjectVertexTriangle(minvv, minvf, xvt, xft, xf, atildec, lambdac, xv))
            return;

        // 3. Update global positions
        // lambda[c] = lambdac;
        SetParticlePosition(vv, xv, x);
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
    __device__ void operator()(GpuIndex i)
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