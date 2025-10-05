/**
 * @file Newton.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for Newton's method for optimization.
 * @date 2025-05-07
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_MATH_OPTIMIZATION_NEWTON_H
#define PBAT_MATH_OPTIMIZATION_NEWTON_H

#include "LineSearch.h"
#include "pbat/Aliases.h"

#include <Eigen/Core>
#include <optional>
#include <type_traits>

namespace pbat::math::optimization {

/**
 * @brief Newton's method for optimization
 * @tparam TScalar Scalar type
 */
template <class TScalar = Scalar>
struct Newton
{
    int nMaxIters; ///< Maximum number of iterations for the Newton solver
    TScalar gtol2; ///< Gradient squared norm threshold for convergence
    Eigen::Vector<TScalar, Eigen::Dynamic> dxk; ///< Step direction
    Eigen::Vector<TScalar, Eigen::Dynamic> gk;  ///< Gradient at current iteration

    /**
     * @brief Construct a new Newton optimizer
     * @param nMaxIters Maximum number of iterations for the Newton solver
     * @param gtol Gradient norm threshold for convergence
     * @param n Number of degrees of freedom
     */
    Newton(int nMaxIters = 10, TScalar gtol = TScalar(1e-4), Index n = 0);
    /**
     * @brief Solve the optimization problem using Newton's method
     *
     * @tparam FPrepareDerivatives Callable type with signature
     * `prepareDerivatives(xk) -> void`
     * @tparam FObjective Callable type for the objective function with signature `f(xk) -> fk`
     * @tparam FGradient Callable type for the gradient with signature `g(xk) -> gk`
     * @tparam FHessianInverseProduct Callable type for the Hessian inverse product with signature
     * `Hinv(xk, gk) -> dxk`
     * @tparam TDerivedX Derived type for the input iterate
     * @param prepareDerivatives Derivative (pre)computation function
     * @param f Objective function
     * @param g Gradient function
     * @param Hinv Hessian inverse product function
     * @param xk Current iterate
     * @param lineSearch Optional line search object
     * @return Squared norm of the gradient at the final iterate
     */
    template <
        class FPrepareDerivatives,
        class FObjective,
        class FGradient,
        class FHessianInverseProduct,
        class TDerivedX>
    TScalar Solve(
        FPrepareDerivatives prepareDerivatives,
        FObjective f,
        FGradient g,
        FHessianInverseProduct Hinv,
        Eigen::MatrixBase<TDerivedX>& xk,
        std::optional<BackTrackingLineSearch<TScalar>> lineSearch = std::nullopt);
};

template <class TScalar>
inline Newton<TScalar>::Newton(int nMaxItersIn, TScalar gtol, Index n)
    : nMaxIters(nMaxItersIn), gtol2(gtol * gtol), dxk(n), gk(n)
{
}

template <class TScalar>
template <
    class FPrepareDerivatives,
    class FObjective,
    class FGradient,
    class FHessianInverseProduct,
    class TDerivedX>
inline TScalar Newton<TScalar>::Solve(
    FPrepareDerivatives prepareDerivatives,
    FObjective f,
    FGradient g,
    FHessianInverseProduct Hinv,
    Eigen::MatrixBase<TDerivedX>& xk,
    std::optional<BackTrackingLineSearch<TScalar>> lineSearch)
{
    TScalar gnorm2{0};
    prepareDerivatives(xk);
    gk = g(xk);
    for (auto k = 0; k < nMaxIters; ++k)
    {
        gnorm2 = gk.squaredNorm();
        if (gnorm2 < gtol2)
            break;
        dxk = -Hinv(xk, gk);
        if (lineSearch)
            lineSearch->Solve(f, gk, dxk, xk);
        else
            xk += dxk;
        prepareDerivatives(xk);
        gk = g(xk);
    }
    return gnorm2;
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_NEWTON_H
