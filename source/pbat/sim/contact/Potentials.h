/**
 * @file Potentials.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for contact potential functions and their derivatives.
 * @version 0.1
 * @date 2025-11-07
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_CONTACT_POTENTIALS_H
#define PBAT_SIM_CONTACT_POTENTIALS_H

#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace pbat::sim::contact::potentials {

namespace mini = math::linalg::mini;

/**
 * @brief Quadratic penalty contact potential and its derivatives.
 *
 * Given a signed distance function \f$ \phi(\mathbf{x}) \f$ where negative distances indicate
 * penetration, as well as a contact radius \f$ r \f$ and penalty stiffness \f$ k_c \f$, the
 * quadratic penalty contact potential is defined as
 * \f[
 * E_\text{penalty}(\phi(\mathbf{x})) = \frac{1}{2} k_c (\phi(\mathbf{x}) - r)^2
 * \f]
 * for \f$ \phi(\mathbf{x}) < r \f$, and zero otherwise.
 *
 * @tparam TScalar Scalar type
 * @tparam nDerivs Number of derivatives to compute (0, 1, or 2)
 * @param sd Signed distance
 * @param kc Penalty stiffness
 * @param r Contact radius
 * @return `|# derivs + 1| x 1` contact potential and its derivatives up to order `nDerivs`
 */
template <int nDerivs, common::CFloatingPoint TScalar>
PBAT_HOST_DEVICE auto SignedDistanceQuadraticPenalty(TScalar sd, TScalar kc, TScalar r = TScalar(0))
    -> mini::SVector<TScalar, nDerivs + 1>
{
    static_assert(nDerivs >= 0 and nDerivs <= 2, "Number of derivatives must be 0, 1, or 2.");
    mini::SVector<TScalar, nDerivs + 1> dE;
    TScalar rd = sd - r;
    if constexpr (nDerivs >= 0)
    {
        dE(0) = TScalar(0.5) * kc * rd * rd;
    }
    if constexpr (nDerivs >= 1)
    {
        dE(1) = kc * rd;
    }
    if constexpr (nDerivs >= 2)
    {
        dE(2) = kc;
    }
    return dE;
}

/**
 * @brief C2 continuous 2-stage activation function from \cite chen_offset_2025
 *
 * @note We expose `kcp, b` as parameters for efficiency, since they can be precomputed given `r`
 * and `kc`. See the preconditions for their definitions.
 *
 * @tparam TScalar Scalar type
 * @param d Distance
 * @param r Contact radius
 * @param kc Quadratic penalty stiffness
 * @param kcp Log barrier stiffness
 * @param b Log barrier offset
 * @return The contact potential
 * @pre `0 <= d <= r`
 * @pre `kc > 0`
 * @pre `kcp = tau*kc*(tau - r)^2`, where `tau = r/2`
 * @pre `b = kc/2*(r - tau)^2 + kcp*log(tau)`, where `tau = r/2`
 */
template <int nDerivs, common::CFloatingPoint TScalar>
PBAT_HOST_DEVICE auto
QuadraticToLogBarrierTwoStageActivation(TScalar d, TScalar r, TScalar kc, TScalar kcp, TScalar b)
    -> mini::SVector<TScalar, nDerivs + 1>
{
    static_assert(nDerivs >= 0 and nDerivs <= 2, "Number of derivatives must be 0, 1, or 2.");
    mini::SVector<TScalar, nDerivs + 1> dE;
    // d >= tau -> d >= r/2 -> 2d >= r
    if (2 * d >= r)
    {
        TScalar rd   = r - d;
        TScalar kcrd = kc * rd;
        if constexpr (nDerivs >= 0)
        {
            dE(0) = TScalar(0.5) * kcrd * rd;
        }
        if constexpr (nDerivs >= 1)
        {
            dE(1) = -kcrd;
        }
        if constexpr (nDerivs >= 2)
        {
            dE(2) = kc;
        }
    }
    else
    {
        using namespace std;
        if constexpr (nDerivs >= 0)
        {
            dE(0) = -kcp * log(d) + b;
        }
        if constexpr (nDerivs >= 1)
        {
            dE(1) = -kcp / d;
        }
        if constexpr (nDerivs >= 2)
        {
            dE(2) = kcp / (d * d);
        }
    }
    return dE;
}

/**
 * @brief Lagged (C1) friction potential from IPC \cite li2020ipc
 * @note We do not include the tangential sliding basis \f$ T_k \f$ here for generality. Obtain the
 * gradient and hessians w.r.t. degrees of freedom via chain ruling (i.e. pre-multiplying gradient
 * w.r.t. `uk` by `Tk` and sandwiching the hessian w.r.t. `uk` as `Tk*Hu*Tk.Transpose()`).
 */
class LaggedFriction
{
  private:
    /**
     * @brief Refer to
     * https://github.com/ipc-sim/ipc-toolkit/blob/v1.4.0/src/ipc/friction/smooth_friction_mollifier.cpp#L8
     * @tparam TScalar
     * @param y
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @return \f$ f_0(y, \epsilon_v h) \f$
     */
    template <common::CFloatingPoint TScalar>
    PBAT_HOST_DEVICE TScalar f0(TScalar y, TScalar epsvh);
    /**
     * @brief Refer to
     * https://github.com/ipc-sim/ipc-toolkit/blob/v1.4.0/src/ipc/friction/smooth_friction_mollifier.cpp#L17
     *
     * @tparam TScalar
     * @param y
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @return \f$ f_1(y, \epsilon_v h) \f$
     */
    template <common::CFloatingPoint TScalar>
    PBAT_HOST_DEVICE TScalar f1(TScalar y, TScalar epsvh);
    /**
     * @brief Refer to
     * https://github.com/ipc-sim/ipc-toolkit/blob/v1.4.0/src/ipc/friction/smooth_friction_mollifier.cpp#L28
     *
     * @tparam TScalar
     * @param y
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @return \f$ \frac{d}{dy} f_1(y, \epsilon_v h) \f$
     */
    template <common::CFloatingPoint TScalar>
    PBAT_HOST_DEVICE TScalar f2(TScalar y, TScalar epsvh);
    /**
     * @brief Refer to
     * https://github.com/ipc-sim/ipc-toolkit/blob/v1.4.0/src/ipc/friction/smooth_friction_mollifier.cpp#L37
     *
     * @tparam TScalar
     * @param y
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @return \f$ f_1(y, \epsilon_v h) / y \f$
     */
    template <common::CFloatingPoint TScalar>
    PBAT_HOST_DEVICE TScalar f1_over_x(const TScalar y, const TScalar epsvh);
    /**
     * @brief Refer to
     * https://github.com/ipc-sim/ipc-toolkit/blob/v1.4.0/src/ipc/friction/smooth_friction_mollifier.cpp#L46
     *
     * @tparam TScalar
     * @param y
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @return \f$ \frac{\left[ \frac{d}{dy} f_1(y, epsvh) - f_1(y, epsvh) \right]}{y^3} \f$
     */
    template <common::CFloatingPoint TScalar>
    PBAT_HOST_DEVICE TScalar f2_x_minus_f1_over_x3(const TScalar y, const TScalar epsvh);

  public:
    /**
     * @brief Evaluate the lagged friction potential w.r.t. sliding velocity \f$ u_k \f$.
     *
     * @tparam TMatrixUk Matrix type for tangential relative velocity
     * @tparam TScalar Scalar type
     * @param uk `2 x 1` tangential relative velocity
     * @param mu Friction coefficient
     * @param lambdakn Normal contact force magnitude
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @return The lagged friction potential
     */
    template <
        mini::CMatrix TMatrixUk,
        common::CFloatingPoint TScalar = typename TMatrixUk::ScalarType>
    PBAT_HOST_DEVICE static TScalar
    Eval(TMatrixUk const& uk, TScalar mu, TScalar lambdakn, TScalar epsvh)
    {
        TScalar ukn = Norm(uk);
        return mu * lambdakn * f0(ukn, epsvh);
    }
    /**
     * @brief Evaluate the lagged friction potential w.r.t. sliding velocity \f$ u_k \f$ and its
     * gradient.
     *
     * @tparam TMatrixUk Matrix type for tangential relative velocity
     * @tparam TMatrixGk Gradient matrix type
     * @tparam TScalar Scalar type
     * @param uk `2 x 1` tangential relative velocity
     * @param mu Friction coefficient
     * @param lambdakn Normal contact force magnitude
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @param gk `2 x 1` Gradient matrix
     * @return The lagged friction potential
     */
    template <
        mini::CMatrix TMatrixUk,
        mini::CMatrix TMatrixGk,
        common::CFloatingPoint TScalar = typename TMatrixUk::ScalarType>
    PBAT_HOST_DEVICE static TScalar
    EvalWithGrad(TMatrixUk const& uk, TScalar mu, TScalar lambdakn, TScalar epsvh, TMatrixGk& gk)
    {
        TScalar ukn      = Norm(uk);
        TScalar mulambda = mu * lambdakn;
        gk = (mulambda * f1_over_x(ukn + std::numeric_limits<TScalar>::epsilon(), epsvh)) * uk;
        return mulambda * f0(ukn, epsvh);
    }
    /**
     * @brief Compute the gradient w.r.t. sliding velocity \f$ u_k \f$.
     *
     * @tparam TMatrixUk Matrix type for tangential relative velocity
     * @tparam TMatrixGk Gradient matrix type
     * @tparam TScalar Scalar type
     * @param uk `2 x 1` tangential relative velocity
     * @param mu Friction coefficient
     * @param lambdakn Normal contact force magnitude
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @param gk `2 x 1` Gradient matrix
     */
    template <
        mini::CMatrix TMatrixUk,
        mini::CMatrix TMatrixGk,
        common::CFloatingPoint TScalar = typename TMatrixUk::ScalarType>
    PBAT_HOST_DEVICE void
    Grad(TMatrixUk const& uk, TScalar mu, TScalar lambdakn, TScalar epsvh, TMatrixGk& gk)
    {
        TScalar ukn = Norm(uk) + std::numeric_limits<TScalar>::epsilon();
        gk          = (mu * lambdakn * f1_over_x(ukn, epsvh)) * uk;
    }
    /**
     * @brief Compute the gradient and Hessian w.r.t. sliding velocity \f$ u_k \f$.
     *
     * @tparam TMatrixUk Matrix type for tangential relative velocity
     * @tparam TMatrixGk Gradient matrix type
     * @tparam TMatrixHk Hessian matrix type
     * @tparam TScalar Scalar type
     * @param uk `2 x 1` tangential relative velocity
     * @param mu Friction coefficient
     * @param lambdakn Normal contact force magnitude
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @param gk `2 x 1` Gradient matrix
     * @param Hk `2 x 2` Hessian matrix
     */
    template <
        mini::CMatrix TMatrixUk,
        mini::CMatrix TMatrixGk,
        mini::CMatrix TMatrixHk,
        common::CFloatingPoint TScalar = typename TMatrixUk::ScalarType>
    PBAT_HOST_DEVICE void GradAndHessian(
        TMatrixUk const& uk,
        TScalar mu,
        TScalar lambdakn,
        TScalar epsvh,
        TMatrixGk& gk,
        TMatrixHk& Hk)
    {
        TScalar ukn      = Norm(uk) + std::numeric_limits<TScalar>::epsilon();
        TScalar mulambda = mu * lambdakn;
        gk               = ((mulambda * f1_over_x(ukn, epsvh))) * uk;
        mini::Identity<TScalar, 2, 2> I;
        Hk = mulambda *
             (f2_x_minus_f1_over_x3(ukn, epsvh) * uk * uk.Transpose() + f1_over_x(ukn, epsvh) * I);
    }
    /**
     * @brief Compute the Hessian w.r.t. sliding velocity \f$ u_k \f$.
     *
     * @tparam TMatrixTk Matrix type for tangential basis
     * @tparam TMatrixUk Matrix type for tangential relative velocity
     * @tparam TMatrixHk Hessian matrix type
     * @tparam TScalar Scalar type
     * @param Tk `2 x 2` Tangential basis matrix
     * @param uk `2 x 1` Tangential relative velocity
     * @param mu Friction coefficient
     * @param lambdakn Normal contact force magnitude
     * @param epsvh \f$ \epsilon_v h \f$ where \f$ \epsilon_v \f$ is IPC's relative velocity
     * threshold for static to dynamic friction's smooth transition, and \f$ h \f$ is the time step
     * @param Hk `2 x 2` Hessian matrix
     */
    template <
        mini::CMatrix TMatrixTk,
        mini::CMatrix TMatrixUk,
        mini::CMatrix TMatrixHk,
        common::CFloatingPoint TScalar = typename TMatrixUk::ScalarType>
    PBAT_HOST_DEVICE void Hessian(
        TMatrixTk const& Tk,
        TMatrixUk const& uk,
        TScalar mu,
        TScalar lambdakn,
        TScalar epsvh,
        TMatrixHk& Hk)
    {
        TScalar ukn = Norm(uk) + std::numeric_limits<TScalar>::epsilon();
        mini::Identity<TScalar, 2, 2> I;
        Hk = (mu * lambdakn) *
             (f2_x_minus_f1_over_x3(ukn, epsvh) * uk * uk.Transpose() + f1_over_x(ukn, epsvh) * I);
    }
};

template <common::CFloatingPoint TScalar>
PBAT_HOST_DEVICE TScalar LaggedFriction::f0(TScalar y, TScalar epsvh)
{
    assert(epsvh > 0);
    assert(y >= 0);
    bool bSliding = y >= epsvh;
    return (bSliding)*y + (not bSliding) * (y * y * (1 - y / (3 * epsvh)) / epsvh + epsvh / 3);
}

template <common::CFloatingPoint TScalar>
PBAT_HOST_DEVICE TScalar LaggedFriction::f1(TScalar y, TScalar epsvh)
{
    assert(epsvh > 0);
    assert(y >= 0);
    bool bSliding              = y >= epsvh;
    const TScalar y_over_eps_v = y / epsvh;
    return (bSliding) * 1 + (not bSliding) * (y_over_eps_v * (2 - y_over_eps_v));
}

template <common::CFloatingPoint TScalar>
PBAT_HOST_DEVICE TScalar LaggedFriction::f2(TScalar y, TScalar epsvh)
{
    assert(epsvh > 0);
    assert(y >= 0);
    bool bSliding = y >= epsvh;
    return (bSliding) * 0 + (not bSliding) * (2 - 2 * y / epsvh) / epsvh;
}

template <common::CFloatingPoint TScalar>
PBAT_HOST_DEVICE TScalar LaggedFriction::f1_over_x(const TScalar y, const TScalar epsvh)
{
    assert(epsvh > 0);
    assert(y >= 0);
    bool bSliding = y >= epsvh;
    return (bSliding) * (1 / y) + (not bSliding) * ((2 - y / epsvh) / epsvh);
}

template <common::CFloatingPoint TScalar>
PBAT_HOST_DEVICE TScalar LaggedFriction::f2_x_minus_f1_over_x3(const TScalar y, const TScalar epsvh)
{
    assert(epsvh > 0);
    assert(y >= 0);
    bool bSliding = y >= epsvh;
    return (bSliding) * (-1 / (y * y * y)) + (not bSliding) * (-1 / (y * epsvh * epsvh));
}

/**
 * @brief Compute gradient with respect to closest points given distance and first energy
 * derivative.
 *
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point
 * @tparam TScalar Scalar type
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point
 * @param d Distance (in the 2-norm) between closest points
 * @param dEdd Derivative of energy with respect to distance
 * @return `2*|# dims| x 1` gradient with respect to closest points `x` and `y`
 */
template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto
GradientWrtClosestPoints(TMatrixX const& x, TMatrixY const& y, TScalar d, TScalar dEdd)
    -> mini::SVector<TScalar, 2 * TMatrixX::kRows>
{
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same dimensions.");
    auto constexpr kDims = TMatrixX::kRows;
    mini::SVector<TScalar, 2 * kDims> g;
    auto gx = g.template Slice<kDims, 1>(0, 0);
    auto gy = g.template Slice<kDims, 1>(kDims, 0);
    gx      = dEdd * (x - y) / (d + std::numeric_limits<TScalar>::min());
    gy      = -gx;
    return g;
}

/**
 * @brief Compute gradient with respect to one of the closest points given distance and first energy
 * derivative.
 *
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point
 * @tparam TScalar Scalar type
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point
 * @param d Distance (in the 2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param i Index indicating which closest point to compute gradient for (0 for `x`, 1 for `y`)
 * @return `|# dims| x 1` gradient with respect to closest point `x` if `i==0`, or `y` if `i==1`
 * @pre `i` must be either `0` or `1`
 */
template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto GradientSegmentWrtClosestPoints(
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    int i) -> mini::SVector<TScalar, TMatrixX::kRows>
{
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same dimensions.");
    auto constexpr kDims = TMatrixX::kRows;
    int const sgn        = (i == 0) * 1 + (i == 1) * -1;
    mini::SVector<TScalar, kDims> gi =
        (sgn * dEdd) * (x - y) / (d + std::numeric_limits<TScalar>::min());
    return gi;
}

/**
 * @brief Compute Hessian with respect to closest points given distance and energy's first and
 * second derivatives.
 *
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point
 * @tparam TScalar Scalar type
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point
 * @param d Distance (in the 2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param d2Edd2 Second derivative of energy with respect to distance
 * @return `2*|# dims| x 2*|# dims|` Hessian with respect to closest points `x` and `y`
 */
template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto HessianWrtClosestPoints(
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    TScalar d2Edd2) -> mini::SMatrix<TScalar, 2 * TMatrixX::kRows, 2 * TMatrixX::kRows>
{
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same dimensions.");
    auto constexpr kDims = TMatrixX::kRows;
    TScalar dinv         = TScalar(1) / (d + std::numeric_limits<TScalar>::min());
    mini::Identity<TScalar, kDims, kDims> I{};
    mini::SVector<TScalar, kDims> const gx            = (x - y) * dinv;
    mini::SMatrix<TScalar, kDims, kDims> const gxgxT  = gx * gx.Transpose();
    mini::SMatrix<TScalar, kDims, kDims> const d2ddxx = dinv * (I - gxgxT);
    mini::SMatrix<TScalar, 2 * kDims, 2 * kDims> H;
    auto Hxx = H.template Slice<kDims, kDims>(0, 0);
    auto Hxy = H.template Slice<kDims, kDims>(0, kDims);
    auto Hyx = H.template Slice<kDims, kDims>(kDims, 0);
    auto Hyy = H.template Slice<kDims, kDims>(kDims, kDims);
    Hxx      = d2Edd2 * gxgxT + dEdd * d2ddxx;
    Hxy      = -Hxx;
    Hyx      = Hxy;
    Hyy      = Hxx;
    return H;
}

/**
 * @brief Compute Hessian block (i,j) with respect to the closest points given distance and energy's
 * first and second derivatives.
 *
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point
 * @tparam TMatrixX::ScalarType Scalar type
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point
 * @param d Distance (in the 2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param d2Edd2 Second derivative of energy with respect to distance
 * @param i Block row index (0 for `x`, 1 for `y`)
 * @param j Block column index (0 for `x`, 1 for `y`)
 * @return `|# dims| x |# dims|` Hessian block `(i,j)` with respect to closest points `x` and `y`
 */
template <
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto HessianBlockWrtClosestPoints(
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    TScalar d2Edd2,
    int i,
    int j) -> mini::SMatrix<TScalar, TMatrixX::kRows, TMatrixX::kRows>
{
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same dimensions.");
    auto constexpr kDims = TMatrixX::kRows;
    TScalar dinv         = TScalar(1) / (d + std::numeric_limits<TScalar>::min());
    mini::Identity<TScalar, kDims, kDims> I{};
    mini::SVector<TScalar, kDims> const gx            = (x - y) * dinv;
    mini::SMatrix<TScalar, kDims, kDims> const gxgxT  = gx * gx.Transpose();
    mini::SMatrix<TScalar, kDims, kDims> const d2ddxx = dinv * (I - gxgxT);
    int const sgn                                     = (i == j) * 1 + (i != j) * -1;
    mini::SMatrix<TScalar, kDims, kDims> Hij = (sgn * d2Edd2) * gxgxT + (sgn * dEdd) * d2ddxx;
    return Hij;
}

/**
 * @brief Compute gradient with respect to linearly interpolated closest points `x = U*a` and `y =
 * V*b`, given distance and first energy derivative.
 *
 * @tparam TMatrixA Matrix type for interpolation weights of first closest point
 * @tparam TMatrixB Matrix type for interpolation weights of second closest point
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point
 * @tparam TScalar Scalar type
 * @param a `|# verts 1| x 1` interpolation weights for first closest point
 * @param b `|# verts 2| x 1` interpolation weights for second closest point
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point
 * @param d Distance (in the 2-norm) between closest points
 * @param dEdd Derivative of energy with respect to distance
 * @return `|# verts 1 * # dims * # verts 2 * # dims| x 1` gradient with respect to `a,b` s.t.
 * closest points `x=U*a` and `y=V*b`
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto GradientWrtLinearlyInterpolatedClosestPoints(
    TMatrixA const& a,
    TMatrixB const& b,
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd)
    -> mini::SVector<TScalar, TMatrixX::kRows * TMatrixA::kRows + TMatrixY::kRows * TMatrixB::kRows>
{
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same number of rows.");
    auto constexpr kDims                  = TMatrixX::kRows;
    mini::SVector<TScalar, 2 * kDims> gxy = GradientWrtClosestPoints(x, y, d, dEdd);
    auto constexpr kUVerts                = TMatrixA::kRows;
    auto constexpr kVVerts                = TMatrixB::kRows;
    auto constexpr kDofsU                 = kDims * kUVerts;
    auto constexpr kDofsV                 = kDims * kVVerts;
    auto constexpr kDofs                  = kDofsU + kDofsV;
    mini::SVector<TScalar, kDofs> guv;
    guv.SetZero();
    auto gu = guv.template Slice<kDofsU, 1>(0, 0);
    auto gv = guv.template Slice<kDofsV, 1>(gu.Rows(), 0);
    pbat::common::ForRange<0, kUVerts>([&]<auto i>() {
        gu.template Slice<kDims, 1>(i * kDims, 0) = gxy.template Slice<kDims, 1>(0, 0) * a(i);
    });
    pbat::common::ForRange<0, kVVerts>([&]<auto i>() {
        gv.template Slice<kDims, 1>(i * kDims, 0) = gxy.template Slice<kDims, 1>(kDims, 0) * b(i);
    });
    return guv;
}

/**
 * @brief Compute Hessian with respect to linearly interpolated closest points `x = U*a` and `y =
 * V*b`, given distance and energy's first and second derivatives.
 * @tparam TMatrixA Matrix type for interpolation weights of first closest point
 * @tparam TMatrixB Matrix type for interpolation weights of second closest point
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point
 * @tparam TScalar Scalar type
 * @param a `|# verts 1| x 1` interpolation weights for first closest point
 * @param b `|# verts 2| x 1` interpolation weights for second closest point
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point
 * @param d Distance (in the 2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param d2Edd2 Second derivative of energy with respect to distance
 * @return `|# verts 1 * # dims * # verts 2 * # dims| x |# verts 1 * # dims * # verts 2 * # dims|`
 * Hessian with respect `a,b` s.t. closest points `x=U*a` and `y=V*b`
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto HessianWrtLinearlyInterpolatedClosestPoints(
    TMatrixA const& a,
    TMatrixB const& b,
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    TScalar d2Edd2)
    -> mini::SMatrix<
        TScalar,
        TMatrixX::kRows * TMatrixA::kRows + TMatrixY::kRows * TMatrixB::kRows,
        TMatrixX::kRows * TMatrixA::kRows + TMatrixY::kRows * TMatrixB::kRows>
{
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same number of rows.");
    static int constexpr kDims = TMatrixX::kRows;
    mini::SMatrix<TScalar, 2 * kDims, 2 * kDims> Hxy =
        HessianWrtClosestPoints(x, y, d, dEdd, d2Edd2);
    static int constexpr kUVerts = TMatrixA::kRows;
    static int constexpr kVVerts = TMatrixB::kRows;
    int constexpr kDofsU         = kDims * kUVerts;
    int constexpr kDofsV         = kDims * kVVerts;
    int constexpr kDofs          = kDofsU + kDofsV;
    mini::SMatrix<TScalar, kDofs, kDofs> H;
    H.SetZero();
    // Compute H block-by-block
    auto Huu = H.template Slice<kDofsU, kDofsU>(0, 0);
    pbat::common::ForRange<0, kUVerts>([&]<int i>() {
        pbat::common::ForRange<0, kUVerts>([&]<int j>() {
            Huu.template Slice<kDims, kDims>(i * kDims, j * kDims) =
                Hxy.template Slice<kDims, kDims>(0, 0) * a(i) * a(j);
        });
    });
    auto Huv = H.template Slice<kDofsU, kDofsV>(0, kDofsU);
    pbat::common::ForRange<0, kUVerts>([&]<int i>() {
        pbat::common::ForRange<0, kVVerts>([&]<int j>() {
            Huv.template Slice<kDims, kDims>(i * kDims, j * kDims) =
                Hxy.template Slice<kDims, kDims>(0, kDims) * a(i) * b(j);
        });
    });
    auto Hvu = H.template Slice<kDofsV, kDofsU>(kDofsU, 0);
    Hvu      = Huv.Transpose();
    auto Hvv = H.template Slice<kDofsV, kDofsV>(kDofsU, kDofsU);
    pbat::common::ForRange<0, kVVerts>([&]<int i>() {
        pbat::common::ForRange<0, kVVerts>([&]<int j>() {
            Hvv.template Slice<kDims, kDims>(i * kDims, j * kDims) =
                Hxy.template Slice<kDims, kDims>(kDims, kDims) * b(i) * b(j);
        });
    });
    return H;
}

/**
 * @brief Compute gradient with respect to one of the linearly interpolated closest points `x = U*a`
 * and `y = V*b`, given distance and first energy derivative.
 *
 * @tparam TMatrixA Matrix type for interpolation weights of first closest point
 * @tparam TMatrixB Matrix type for interpolation weights of second closest point
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point
 * @tparam TScalar Scalar type
 * @param a `|# verts 1| x 1` interpolation weights of first closest point
 * @param b `|# verts 1| x 1` interpolation weights of second closest point
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point
 * @param d Distance (in the 2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param ib Index indicating which closest point to compute gradient for (0 for `x`, 1 for `y`)
 * @param i Index indicating which vertex of the selected closest point to compute gradient for
 * @return `|# dims| x 1` gradient with respect to vertex `i` of closest point `x` if `ib==0`, or
 * `y` if `ib==1`
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto GradientSegmentWrtLinearlyInterpolatedClosestPoints(
    TMatrixA const& a,
    TMatrixB const& b,
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    int ib,
    int i) -> mini::SVector<TScalar, TMatrixX::kRows>
{
    TScalar gammai = (ib == 0) * a(i) + (ib != 0) * b(i);
    return gammai * GradientSegmentWrtClosestPoints(x, y, d, dEdd, ib);
}

/**
 * @brief Compute Hessian block (i,j) with respect to the linearly interpolated closest points `x =
 * U*a` and `y = V*b`, given distance and energy's first and second derivatives.
 *
 * @tparam TMatrixA Matrix type for interpolation weights of first closest point
 * @tparam TMatrixB Matrix type for interpolation weights of second closest point
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point
 * @tparam TScalar Scalar type
 * @param a `|# verts 1| x 1` interpolation weights for first closest point
 * @param b `|# verts 2| x 1` interpolation weights for second closest point
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point
 * @param d Distance (in the 2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param d2Edd2 Second derivative of energy with respect to distance
 * @param ib Index indicating which closest point to compute Hessian block row for (0 for `x`, 1 for
 * `y`)
 * @param jb Index indicating which closest point to compute Hessian block column for (0 for `x`, 1
 * for `y`)
 * @param i Index indicating which vertex of the selected closest point for block row to compute
 * Hessian block row for
 * @param j Index indicating which vertex of the selected closest point for block column to compute
 * Hessian block column for
 * @return `|# dims| x |# dims|` Hessian block `(i,j)` with respect to vertex `i` of closest point
 * `x` if `ib==0`, or `y` if `ib==1`, and vertex `j` of closest point `x` if `jb==0`, or `y` if
 * `jb==1`
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixB,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto HessianBlockWrtLinearlyInterpolatedClosestPoints(
    TMatrixA const& a,
    TMatrixB const& b,
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    TScalar d2Edd2,
    int ib,
    int jb,
    int i,
    int j) -> mini::SMatrix<TScalar, TMatrixX::kRows, TMatrixX::kRows>
{
    TScalar gammai = (ib == 0) * a(i) + (ib != 0) * b(i);
    TScalar gammaj = (jb == 0) * a(j) + (jb != 0) * b(j);
    return (gammai * gammaj) * HessianBlockWrtClosestPoints(x, y, d, dEdd, d2Edd2, ib, jb);
}

/**
 * @brief Compute gradient with respect to linearly interpolated closest point x = U*a when y is
 * static.
 *
 * This overload computes the gradient only with respect to the interpolation weights a of the
 * first closest point x. The second closest point y is treated as a fixed (static) point, so no
 * derivatives with respect to y (or any interpolation weights for y) are formed.
 *
 * @tparam TMatrixA Matrix type for interpolation weights of first closest point
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point (static)
 * @tparam TScalar Scalar type
 * @param a `|# verts 1| x 1` interpolation weights for first closest point
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point (static)
 * @param d Distance (2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @return `|# verts 1 * # dims| x 1` gradient with respect to `a` such that `x = U*a` and `y` is
 * static
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto GradientWrtLinearlyInterpolatedClosestPoints(
    TMatrixA const& a,
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd) -> mini::SVector<TScalar, TMatrixX::kRows * TMatrixA::kRows>
{
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same number of rows.");
    auto constexpr kDims                  = TMatrixX::kRows;
    auto constexpr kUVerts                = TMatrixA::kRows;
    auto constexpr kDofsU                 = kDims * kUVerts;
    mini::SVector<TScalar, 2 * kDims> gxy = GradientWrtClosestPoints(x, y, d, dEdd);
    auto gx                               = gxy.template Slice<kDims, 1>(0, 0);
    mini::SVector<TScalar, kDofsU> gu;
    pbat::common::ForRange<0, kUVerts>(
        [&]<auto i>() { gu.template Slice<kDims, 1>(i * kDims, 0) = gx * a(i); });
    return gu;
}

/**
 * @brief Compute Hessian with respect to linearly interpolated closest point x = U*a when y is
 * static.
 *
 * This overload computes the Hessian only with respect to the interpolation weights a of the first
 * closest point x. The second closest point y is treated as static.
 *
 * @tparam TMatrixA Matrix type for interpolation weights of first closest point
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point (static)
 * @tparam TScalar Scalar type
 * @param a `|# verts 1| x 1` interpolation weights for first closest point
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point (static)
 * @param d Distance (2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param d2Edd2 Second derivative of energy with respect to distance
 * @return `|# verts 1 * # dims| x |# verts 1 * # dims|` Hessian with respect to `a` such that
 * `x=U*a` and `y` is static
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto HessianWrtLinearlyInterpolatedClosestPoints(
    TMatrixA const& a,
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    TScalar d2Edd2)
    -> mini::SMatrix<TScalar, TMatrixX::kRows * TMatrixA::kRows, TMatrixX::kRows * TMatrixA::kRows>
{
    static_assert(TMatrixX::kRows == TMatrixY::kRows, "x and y must have the same number of rows.");
    auto constexpr kDims   = TMatrixX::kRows;
    auto constexpr kUVerts = TMatrixA::kRows;
    auto constexpr kDofsU  = kDims * kUVerts;
    mini::SMatrix<TScalar, 2 * kDims, 2 * kDims> const Hxy =
        HessianWrtClosestPoints(x, y, d, dEdd, d2Edd2);
    auto Hxx = Hxy.template Slice<kDims, kDims>(0, 0);
    mini::SMatrix<TScalar, kDofsU, kDofsU> H;
    pbat::common::ForRange<0, kUVerts>([&]<auto i>() {
        pbat::common::ForRange<0, kUVerts>([&]<auto j>() {
            H.template Slice<kDims, kDims>(i * kDims, j * kDims) = Hxx * a(i) * a(j);
        });
    });
    return H;
}

/**
 * @brief Compute gradient segment for vertex i of the linearly interpolated closest point x = U*a
 * when y is static.
 *
 * This overload computes the gradient contribution associated with the i-th vertex of x only.
 *
 * @tparam TMatrixA Matrix type for interpolation weights of first closest point
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point (static)
 * @tparam TScalar Scalar type
 * @param a `|# verts 1| x 1` interpolation weights for first closest point
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point (static)
 * @param d Distance (2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param i Vertex index on x to compute the gradient segment for
 * @return `|# dims| x 1` gradient with respect to vertex `i` of closest point `x`
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto GradientSegmentWrtLinearlyInterpolatedClosestPoints(
    TMatrixA const& a,
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    int i) -> mini::SVector<TScalar, TMatrixX::kRows>
{
    return a(i) * GradientSegmentWrtClosestPoints(x, y, d, dEdd, 0);
}

/**
 * @brief Compute Hessian block (i,j) for the linearly interpolated closest point x = U*a when y is
 * static.
 *
 * This overload computes the Hessian block corresponding to vertices (i,j) of x only.
 *
 * @tparam TMatrixA Matrix type for interpolation weights of first closest point
 * @tparam TMatrixX Matrix type for first closest point
 * @tparam TMatrixY Matrix type for second closest point (static)
 * @tparam TScalar Scalar type
 * @param a `|# verts 1| x 1` interpolation weights for first closest point
 * @param x `|# dims| x 1` first closest point
 * @param y `|# dims| x 1` second closest point (static)
 * @param d Distance (2-norm) between closest points
 * @param dEdd First derivative of energy with respect to distance
 * @param d2Edd2 Second derivative of energy with respect to distance
 * @param i Vertex index on x for block row
 * @param j Vertex index on x for block column
 * @return `|# dims| x |# dims|` Hessian block `(i,j)` with respect to vertices of `x`
 */
template <
    mini::CMatrix TMatrixA,
    mini::CMatrix TMatrixX,
    mini::CMatrix TMatrixY,
    class TScalar = typename TMatrixX::ScalarType>
PBAT_HOST_DEVICE auto HessianBlockWrtLinearlyInterpolatedClosestPoints(
    TMatrixA const& a,
    TMatrixX const& x,
    TMatrixY const& y,
    TScalar d,
    TScalar dEdd,
    TScalar d2Edd2,
    int i,
    int j) -> mini::SMatrix<TScalar, TMatrixX::kRows, TMatrixX::kRows>
{
    return (a(i) * a(j)) * HessianBlockWrtClosestPoints(x, y, d, dEdd, d2Edd2, 0, 0);
}

} // namespace pbat::sim::contact::potentials

#endif // PBAT_SIM_CONTACT_POTENTIALS_H
