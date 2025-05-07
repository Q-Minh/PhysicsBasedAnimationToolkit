/**
 * @file LineSearch.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for line search algorithms.
 * @date 2025-05-07
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_MATH_OPTIMIZATION_LINESEARCH_H
#define PBAT_MATH_OPTIMIZATION_LINESEARCH_H

#include "pbat/Aliases.h"

#include <Eigen/Core>

namespace pbat::math::optimization {

template <class TScalar = Scalar>
struct BackTrackingLineSearch
{
    TScalar alpha{1};       ///< Initial step size
    TScalar const tau{0.5}; ///< Step size decrease factor
    TScalar const c{1e-4};  ///< Armijo slope scale
    int nMaxIters{20};      ///< Maximum number of iterations for the line search

    /**
     * @brief Perform a backtracking line search
     *
     * @tparam FObjective Callable type for the objective function with signature `f(xk) -> fk`
     * @tparam TDerivedG Derived type for the gradient
     * @tparam TDerivedDX Derived type for the step direction
     * @tparam TDerivedX Derived type for the current iterate
     * @param f Objective function
     * @param g Gradient at the initial iterate
     * @param dx Step direction
     * @param xk Current iterate
     * @return Step size
     */
    template <class FObjective, class TDerivedG, class TDerivedDX, class TDerivedX>
    TScalar Solve(
        FObjective f,
        Eigen::DenseBase<TDerivedG> const& g,
        Eigen::DenseBase<TDerivedDX> const& dx,
        Eigen::DenseBase<TDerivedX>& xk) const
    {
        TScalar alphaj    = alpha;
        TScalar const Dfk = g.dot(dx);
        TScalar fk        = f(xk);
        for (auto j = 0; j < nMaxIters; ++j)
        {
            TScalar flinear = fk + (c * alphaj) * Dfk;
            xk              = xk + alphaj * dx;
            fk              = f(xk);
            if (fk <= flinear)
                break;
            alphaj *= tau;
        }
        return alphaj;
    }
};

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_LINESEARCH_H
