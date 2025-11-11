#ifndef PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H
#define PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H

#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Cast.h"
#include "pbat/math/linalg/mini/Inverse.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Norm.h"
#include "pbat/math/linalg/mini/Product.h"
#include "pbat/math/linalg/mini/Reductions.h"
#include "pbat/math/linalg/mini/UnaryOperations.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>

namespace pbat::math::optimization {

/**
 * @brief Parameters for trust-region SR1 optimization in a triangle
 * @tparam TScalar Scalar type
 */
template <common::CFloatingPoint TScalar>
struct TriangleConstrainedTrustRegionSr1Params
{
    /**
     * @brief Read-only parameters
     */

    TScalar R0;   ///< Initial trust region radius
    TScalar eta;  ///< Trust region minimal energy reduction ratio
    TScalar trlo; ///< Largest energy reduction ratio under which to shrink trust region. Must
                  ///< satisfy `0 < trlo < trhi`
    TScalar trhi; ///< Smallest energy reduction ratio over which to grow trust region. Must satisfy
                  ///< `trlo < trhi < 1`
    TScalar trbound;  ///< Smallest step size multiple of trust region radius over which to grow
                      ///< trust region. Must satisfy `0 < trbound <= 1`.
    TScalar trgrow;   ///< Trust region growth factor
    TScalar trshrink; ///< Trust region shrink factor
    TScalar sigmaB;   ///< Initial Hessian approximation scaling
    TScalar delta0;   ///< Numerical offset preventing division by zero when computing the ratio of
                      ///< actual reduction to predicted reduction.
    int nMaxIters;    ///< Maximum number of solver iterations
    TScalar gzero;    ///< Gradient norm convergence tolerance. Must satisfy `gzero > 0`.

    /**
     * @brief Read/Write parameters
     */
    int k;                                      ///< Number of iterations taken
    TScalar Rk;                                 ///< Trust region radius at current iteration
    TScalar fk;                                 ///< Function value at current iteration
    math::linalg::mini::SVector<TScalar, 2> gk; ///< `2 x 1` gradient at current iteration
    math::linalg::mini::SMatrix<TScalar, 2, 2>
        Bk; ///< `2 x 2` Hessian approximation at current iteration
};

/**
 * @brief Minimizes a convex quadratic function \f$ f(x) = \frac{1}{2} (x-x_k)^T B (x-x_k) + g^T
 * (x-x_k) \f$ subject to staying in the reference triangle, i.e.
 * \f$ 0 \leq x \leq 1, \; 1^T x \leq 1 \f$.
 *
 * @tparam TMatrixXk Matrix type for initial guess
 * @tparam TMatrixGk Matrix type for gradient
 * @tparam TMatrixBk Matrix type for Hessian approximation
 * @tparam TScalar Scalar type
 * @param xk `2 x 1` initial guess
 * @param gk `2 x 1` gradient at initial guess
 * @param Bk `2 x 2` hessian approximation at initial guess
 * @return `2 x 1` solution in reference triangle
 */
template <
    math::linalg::mini::CMatrix TMatrixXk,
    math::linalg::mini::CMatrix TMatrixGk,
    math::linalg::mini::CMatrix TMatrixBk,
    class TScalar = typename TMatrixXk::ScalarType>
PBAT_HOST_DEVICE auto SolvePositiveQuadraticInReferenceTriangle(
    TMatrixXk const& xk,
    TMatrixGk const& gk,
    TMatrixBk const& Bk) -> math::linalg::mini::SVector<TScalar, 2>
{
    static_assert(TMatrixXk::kRows == 2 and TMatrixXk::kCols == 1, "xk must be 2 x 1.");
    static_assert(TMatrixGk::kRows == 2 and TMatrixGk::kCols == 1, "gk must be 2 x 1.");
    static_assert(TMatrixBk::kRows == 2 and TMatrixBk::kCols == 2, "Bk must be 2 x 2.");
    namespace mini                  = math::linalg::mini;
    mini::SVector<TScalar, 2> xstar = xk - mini::Inverse(Bk) * gk;
    TScalar constexpr zero{0};
    TScalar constexpr one{1};
    TScalar constexpr half{0.5};
    bool const bFeasible = (xstar(0) >= zero and xstar(1) >= zero) and (xstar(0) + xstar(1) <= one);
    if (not bFeasible)
    {
        // Derivation and CSE by-hand for the quadratic
        // f(t) = 0.5 (x0 + t dx - xk)^T Bk (x0 + t dx - xk) + gk^T (x0 + t dx - xk)
        // where x = x0 + t dx is constrained to one of the triangle edges.
        // Edge 1: x0 = [0,0], dx = [0,1]
        // Edge 2: x0 = [0,0], dx = [1,0]
        // Edge 3: x0 = [0,1], dx = [1,-1]
        TScalar gkTxk                  = Dot(gk, xk);
        mini::SVector<TScalar, 2> Bkxk = Bk * xk;
        TScalar xkTBkxk                = Dot(xk, Bkxk);
        TScalar a1                     = -gkTxk + half * xkTBkxk;
        TScalar b1                     = gk(1) - Bkxk(1);
        TScalar c1                     = Bk(1, 1);
        TScalar a2                     = a1;
        TScalar b2                     = gk(0) - Bkxk(0);
        TScalar c2                     = Bk(0, 0);
        mini::SVector<TScalar, 2> xk0{-xk(0), 1 - xk(1)};
        TScalar a3 = Dot(gk, xk0) + half * Dot(xk0, Bk * xk0);
        TScalar b3 = (gk(0) - gk(1)) + (Bk(0, 1) - Bk(1, 1)) - (Bkxk(0) - Bkxk(1));
        TScalar c3 = Bk(0, 0) - 2 * Bk(0, 1) + Bk(1, 1);
        // Minimize quadratic a_i + b_i t + 1/2 c_i t^2 assuming c_i > 0
        using namespace std;
        TScalar tmin1 = min(max(-b1 / c1, zero), one);
        TScalar tmin2 = min(max(-b2 / c2, zero), one);
        TScalar tmin3 = min(max(-b3 / c3, zero), one);
        mini::SVector<TScalar, 3> fmins{
            a1 + b1 * tmin1 + half * c1 * tmin1 * tmin1,
            a2 + b2 * tmin2 + half * c2 * tmin2 * tmin2,
            a3 + b3 * tmin3 + half * c3 * tmin3 * tmin3};
        // Vectorized argmin
        mini::SVector<int, 3> argmin{
            fmins(0) <= fmins(1) and fmins(0) <= fmins(2),
            fmins(1) <= fmins(0) and fmins(1) <= fmins(2),
            fmins(2) <= fmins(0) and fmins(2) <= fmins(1)};
        mini::SMatrix<TScalar, 2, 3> xstars;
        xstars.Col(0) = mini::SVector<TScalar, 2>{zero, tmin1};
        xstars.Col(1) = mini::SVector<TScalar, 2>{tmin2, zero};
        xstars.Col(2) = mini::SVector<TScalar, 2>{tmin3, one - tmin3};
        xstar         = xstars * argmin / SumReduce(argmin);
    }
    return xstar;
}

/**
 * @brief Check convergence of the triangle constrained minimization.
 *
 * This function checks the convergence of the triangle constrained minimization
 * by verifying the KKT conditions. Recall that the KKT conditions are only first-order
 * necessary conditions for optimality. However, the constraints for a triangle minimization
 * are purely linear, so that the Lagrangian's hessian is actually just the objective
 * function's hessian. Thus, if the hessian of the objective function is positive (semi-)definite
 * and the KKT conditions are satisfied, then we have found a local minimizer, i.e.
 * convergence has been reached.
 *
 * @tparam TMatrixXk Matrix type for current iterate
 * @tparam TMatrixGk Matrix type for current gradient
 * @tparam TScalar Scalar type
 * @param xk `2 x 1` current iterate
 * @param gk `2 x 1` current gradient
 * @param gzero Gradient norm convergence tolerance
 * @return `true` if converged, `false` otherwise
 * @pre `xk(0) >= 0 and xk(1) >= 0 and (xk(0)+xk(1)) <= 1`, i.e. `xk` must be in the feasible set
 * (the reference triangle)
 * @pre The hessian of the objective function is positive semi-definite.
 */
template <
    math::linalg::mini::CMatrix TMatrixXk,
    math::linalg::mini::CMatrix TMatrixGk,
    class TScalar = typename TMatrixXk::ScalarType>
PBAT_HOST_DEVICE bool
CheckTriangleMinimizationConvergence(TMatrixXk const& xk, TMatrixGk const& gk, TScalar gzero)
{
    static_assert(TMatrixXk::kRows == 2 and TMatrixXk::kCols == 1, "xk must be 2 x 1.");
    static_assert(TMatrixGk::kRows == 2 and TMatrixGk::kCols == 1, "gk must be 2 x 1.");
    assert(xk(0) >= TScalar(0) and xk(1) >= TScalar(0) and (xk(0) + xk(1)) <= TScalar(1));
    namespace mini = math::linalg::mini;
    bool bConverged{false};
    // Check convergence via KKT conditions (constraints are linear, so hessian of Lagrangian is
    // Bk if Bk is an accurate estimate of the objective function's hessian)
    mini::SVector<bool, 3> active{
        xk(0) == TScalar(0),
        xk(1) == TScalar(0),
        xk(0) + xk(1) == TScalar(1)};
    int nActive = SumReduce(Cast<int>(active));
    switch (nActive)
    {
        // Recall the constraints are: x(0) >= 0, x(1) >= 0, x(0) + x(1) <= 1, so
        // the constraint gradients are: gradc0 = [1,0], gradc1 = [0,1], gradc2 = [-1,-1].
        // Note also that there can be at most 2 active constraints, since a point in the
        // reference triangle can only be on 2 edges at most (i.e. when it is a vertex).
        // When there are active constraints (i.e. 1 or 2), the strategy is then to solve
        // for the (unique) Lagrange multipliers which satisfy the KKT stationarity condition,
        // and then check their positivity. Feasibility of `xk` is assumed, so that
        // complementarity is automatically satisfied (if active Lagrange multipliers are positive,
        // the associated constraint must be at its bound). We will again vectorize some
        // conditionals. Sorry for the hard-to-read code.
        case 2: {
            int inactiveIdx =
                /*(not active(0)) * 0 + */ (not active(1)) * 1 + (not active(2)) * 2;
            int i = (inactiveIdx + 1) % 3;
            int j = (inactiveIdx + 2) % 3;
            mini::SMatrix<TScalar, 2, 2> gradcij;
            // clang-format off
                gradcij(0,0) = (i == 0)*TScalar(1) + /*(i == 1)*TScalar(0) + */(i == 2)*TScalar(-1);
                gradcij(1,0) = /*(i == 0)*TScalar(0) + */(i == 1)*TScalar(1) + (i == 2)*TScalar(-1);
                gradcij(0,1) = (j == 0)*TScalar(1) + /*(j == 1)*TScalar(0) + */(j == 2)*TScalar(-1);
                gradcij(1,1) = /*(j == 0)*TScalar(0) + */(j == 1)*TScalar(1) + (j == 2)*TScalar(-1);
            // clang-format on
            mini::SVector<TScalar, 2> lambdakij = Inverse(gradcij) * gk;
            bConverged = (lambdakij(0) > TScalar(0)) and (lambdakij(1) > TScalar(0));
            break;
        }
        case 1: {
            int activeIdx = /*active(0) * 0 + */ active(1) * 1 + active(2) * 2;
            mini::SVector<TScalar, 2> gradci{
                active(0) * TScalar(1) /* + active(1)*TScalar(0)*/ - active(2) * TScalar(1),
                /*active(0)*TScalar(0) + */ active(1) * TScalar(1) - active(2) * TScalar(1)};
            TScalar lambdaki = Dot(gradci, gk) / SquaredNorm(gradci);
            bConverged       = lambdaki > TScalar(0);
            break;
        }
        default: {
            bConverged = SquaredNorm(gk) < gzero * gzero;
            break;
        }
    }
    return bConverged;
}

/**
 * @brief Minimize a function \f$ f(x) \f$ subject to \f$ x \in \text{Triangle} \f$ using a
 * trust-region approach with modified SR1 update.
 *
 * We directly optimize in the reference space (i.e. triangle barycentric coordinates `x`), where
 * the triangle constraints are simply \f$ 0 \leq x_i \leq 1, \; \sum_i x_i \leq 1 \f$.
 *
 * @tparam FObjective Callable type with signature `TScalar (TMatrixXk const&)`
 * @tparam FGradient Callable type with signature `TMatrixXk (TMatrixXk const&)`
 * @tparam FCheckConvergence Callable type with signature `bool (TMatrixXk const& xk, bool
 * bStepAccepted)`
 * @tparam TMatrixXk Matrix type for optimization variable
 * @tparam TScalar Scalar type
 * @param f Objective function taking in a `pbat::math::linalg::mini::SVector<TScalar, 2> const&`
 * and returning the objective function value
 * @param gradf Gradient function taking in a `pbat::math::linalg::mini::SVector<TScalar, 2> const&`
 * and returning the `2 x 1` gradient vector
 * @param fCheckConvergence Convergence check function taking in the current iteration number,
 * current iterate, current function value, current gradient, and gradient norm convergence, and
 * returning true if converged
 * @param xk `2 x 1` initial iterate (in barycentric coordinates)
 * @param params Optimization parameters (read/write)
 * @return true if the optimization converged, false otherwise
 */
template <
    class FObjective,
    class FGradient,
    class FCheckConvergence,
    math::linalg::mini::CMatrix TMatrixXk,
    class TScalar = typename TMatrixXk::ScalarType>
bool TriangleConstrainedTrustRegionSr1(
    FObjective const& f,
    FGradient const& gradf,
    FCheckConvergence const& fCheckConvergence,
    TMatrixXk& xk,
    TriangleConstrainedTrustRegionSr1Params<TScalar>& params)
{
    namespace mini = math::linalg::mini;
    static_assert(TMatrixXk::kRows == 2 and TMatrixXk::kCols == 1, "xk must be 2 x 1.");
    // Simplify variable names
    TScalar& Rk                      = params.Rk;
    TScalar& fk                      = params.fk;
    mini::SVector<TScalar, 2>& gk    = params.gk;
    mini::SMatrix<TScalar, 2, 2>& Bk = params.Bk;
    // Initialize solve
    Rk = params.R0;
    fk = f(xk);
    gk = gradf(xk);
    Bk = params.sigmaB * mini::Identity<TScalar, 2, 2>();
    // NOTE: The KKT convergence check is valid for a smooth objective function `f`,
    // but not for functions like SDFs which have discontinuous gradients at the medial axis,
    // and no vanishing gradient. Thus, I want to abstract away the convergence check to the
    // caller, who can implement a more suitable convergence criterion if needed.
    bool bConverged{false};
    for (params.k = 0; params.k < params.nMaxIters and not bConverged; ++params.k)
    {
        mini::SVector<TScalar, 2> xkp1 = SolvePositiveQuadraticInReferenceTriangle(xk, gk, Bk);
        mini::SVector<TScalar, 2> sk   = xkp1 - xk;
        TScalar lensk                  = MaxReduce(Abs(sk));
        bool bTruncateStep             = lensk > Rk;
        // We vectorize the if (bTruncateStep) branch to avoid thread divergence in GPU code
        sk    = (not bTruncateStep) * sk + (bTruncateStep) * (sk * Rk / lensk);
        xkp1  = xk + sk;
        lensk = MaxReduce(Abs(sk));
        mini::SVector<TScalar, 2> gkp1 = gradf(xkp1);
        mini::SVector<TScalar, 2> yk   = gkp1 - gk;
        TScalar fkp1                   = f(xkp1);
        TScalar ared                   = fk - fkp1;
        mini::SVector<TScalar, 2> Bksk = Bk * sk;
        TScalar skTBksk                = Dot(sk, Bksk);
        TScalar mkp1                   = Dot(gk, sk) + TScalar(0.5) * skTBksk;
        TScalar pred                   = -mkp1;
        TScalar rho                    = ared / (pred + params.delta0);
        // We vectorize trust-region update as well
        bool bGrowTrustRegion   = rho > params.trhi and lensk >= params.trbound * Rk;
        bool bShrinkTrustRegion = rho < params.trlo;
        Rk                      = (not bGrowTrustRegion and not bShrinkTrustRegion) * Rk +
             (bGrowTrustRegion) * (params.trgrow * Rk) +
             (bShrinkTrustRegion) * (params.trshrink * Rk);
        // stable = den**2 >= r * np.dot(sk, sk) * np.dot(vk, vk)
        // Bkp1 = Bk + np.outer(vk, vk) / den if stable else Bk
        // Update inverse hessian estimate and keep positive definite
        mini::SVector<TScalar, 2> vk = yk - Bksk;
        TScalar den                  = Dot(vk, sk);
        TScalar skTyk                = Dot(sk, yk);
        bool bStable                 = den != TScalar(0);
        bool bUpdateBk               = skTyk > skTBksk and bStable;
        den += (not bStable) * std::numeric_limits<TScalar>::epsilon();
        Bk += (bUpdateBk) * ((vk * vk.Transpose()) / den);
        // Vectorize step acceptance/rejection
        bool bStepAccepted = rho > params.eta;
        xk                 = (bStepAccepted)*xkp1 + (not bStepAccepted) * xk;
        fk                 = (bStepAccepted)*fkp1 + (not bStepAccepted) * fk;
        gk                 = (bStepAccepted)*gkp1 + (not bStepAccepted) * gk;
        // Check convergence
        bConverged = fCheckConvergence(xk, bStepAccepted);
    }
    return bConverged;
}

/**
 * @brief Minimize a function \f$ f(x) \f$ subject to \f$ x \in \text{Triangle} \f$ using a
 * trust-region approach with modified SR1 update.
 *
 * This overload uses the default KKT condition check for convergence, suitable for smooth problems.
 *
 * @tparam FObjective Callable type with signature `TScalar (TMatrixXk const&)`
 * @tparam FGradient Callable type with signature `TMatrixXk (TMatrixXk const&)`
 * @tparam TMatrixXk Matrix type for optimization variable
 * @tparam TScalar Scalar type
 * @param f Objective function taking in a `pbat::math::linalg::mini::SVector<TScalar, 2> const&`
 * and returning the objective function value
 * @param gradf Gradient function taking in a `pbat::math::linalg::mini::SVector<TScalar, 2> const&`
 * and returning the `2 x 1` gradient vector
 * @param xk `2 x 1` initial iterate (in barycentric coordinates)
 * @param params Optimization parameters (read/write)
 * @return true if the optimization converged, false otherwise
 */
template <
    class FObjective,
    class FGradient,
    math::linalg::mini::CMatrix TMatrixXk,
    class TScalar = typename TMatrixXk::ScalarType>
bool TriangleConstrainedTrustRegionSr1(
    FObjective const& f,
    FGradient const& gradf,
    TMatrixXk& xk,
    TriangleConstrainedTrustRegionSr1Params<TScalar>& params)
{
    return TriangleConstrainedTrustRegionSr1(
        f,
        gradf,
        [&](TMatrixXk const& xk, bool bStepAccepted) {
            return CheckTriangleMinimizationConvergence(xk, params.gk, params.gzero);
        },
        xk,
        params);
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H
