#ifndef PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H
#define PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H

#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Inverse.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Norm.h"
#include "pbat/math/linalg/mini/Product.h"
#include "pbat/math/linalg/mini/Reductions.h"

#include <Eigen/Core>
#include <cmath>

namespace pbat::math::optimization {

/**
 * @brief Parameters for trust-region SR1 optimization in a triangle
 * @tparam TScalar Scalar type
 */
template <common::CFloatingPoint TScalar>
struct MinimizeInTriangleWithTrustRegionSr1Params
{
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
    TScalar
        delta0; ///< Numerical offset preventing division by zero in cases of zero energy reduction.
    int nMaxIters; ///< Maximum number of solver iterations
};

/**
 * @brief Minimizes a convex quadratic function \f$ f(x) = \frac{1}{2} x^T B x + g^T x \f$ subject
 * to staying in the reference triangle, i.e.
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
    bool const bFeasible = (xstar(0) >= zero and xstar(1) >= zero) and
                           (xstar(0) <= one and xstar(1) <= one) and (xstar(0) + xstar(1) <= one);
    if (not bFeasible)
    {
        // Derivation and CSE by-hand for the quadratic
        // f(t) = 0.5 (x0 + t dx - xk)^T Bk (x0 + t dx - xk) + gk^T (x0 + t dx - xk)
        // where x = x0 + t dx is constrained to one of the triangle edges.
        // Edge 1: x0 = [0,0], dx = [0,1]
        // Edge 2: x0 = [0,0], dx = [1,0]
        // Edge 3: x0 = [0,1], dx = [1,-1]
        TScalar gkTxk                  = gk.Transpose() * xk;
        mini::SVector<TScalar, 2> Bkxk = Bk * xk;
        TScalar xkTBkxk                = xk.Transpose() * Bkxk;
        TScalar a1                     = -gkTxk + half * xkTBkxk;
        TScalar b1                     = gk(1) - Bkxk(1);
        TScalar c1                     = Bk(1, 1);
        TScalar a2                     = a1;
        TScalar b2                     = gk(0) - Bkxk(0);
        TScalar c2                     = Bk(0, 0);
        mini::SVector<TScalar, 2> xk0{-xk(0), 1 - xk(1)};
        TScalar a3 = gk.Transpose() * xk0 + half * xk0.Transpose() * Bk * xk0;
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
        xstar         = xstars * argmin / Reduce(argmin);
    }
    return xstar;
}

template <
    class FObjective,
    class FGradient,
    math::linalg::mini::CMatrix TMatrixXk,
    class TScalar = typename TMatrixXk::ScalarType>
bool MinimizeInTriangleWithTrustRegionSr1(
    FObjective const& f,
    FGradient const& gradf,
    TMatrixXk& xk,
    MinimizeInTriangleWithTrustRegionSr1Params<TScalar> const& params)
{
    namespace mini = math::linalg::mini;
    static_assert(TMatrixXk::kRows == 2 and TMatrixXk::kCols == 1, "xk must be 2 x 1.");
    TScalar Rk = params.R0;
    
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H
