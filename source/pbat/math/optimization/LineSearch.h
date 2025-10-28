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

/**
 * @brief Backtracking line search algorithm
 *
 * @tparam TScalar Scalar type
 */
template <class TScalar = Scalar>
struct BackTrackingLineSearch
{
    int nMaxIters{20}; ///< Maximum number of iterations for the line search
    TScalar tau{0.5};  ///< Step size decrease factor
    TScalar c{1e-4};   ///< Armijo slope scale
    TScalar alpha{1};  ///< Initial step size

    TScalar alphaj;                            ///< Current step size
    TScalar fj;                                ///< Current objective function value
    Eigen::Vector<TScalar, Eigen::Dynamic> xj; ///< Current candidate iterate
    int niters;                                ///< Current iteration

    /**
     * @brief Construct a new Back Tracking Line Search object
     *
     * @param nMaxIters Maximum number of iterations for the line search
     * @param tau Step size decrease factor
     * @param c Armijo slope scale
     * @param alpha Initial step size
     * @param n Number of degrees of freedom
     */
    BackTrackingLineSearch(
        int nMaxIters  = 20,
        TScalar tau    = TScalar(0.5),
        TScalar c      = TScalar(1e-4),
        TScalar alpha  = TScalar(1),
        Eigen::Index n = 0);
    /**
     * @brief Perform a backtracking line search
     *
     * @tparam FObjective Callable type for the objective function with signature `f(xk) -> fk`
     * @tparam TDerivedG Derived type for the gradient
     * @tparam TDerivedDX Derived type for the step direction
     * @tparam TDerivedX Derived type for the current iterate
     * @param f Objective function
     * @param fk Objective function value at the current iterate
     * @param gk Gradient at the initial iterate
     * @param dx Step direction
     * @param xk Current iterate
     * @return true if the line search succeeded, false otherwise
     */
    template <class FObjective, class TDerivedG, class TDerivedDX, class TDerivedX>
    bool Solve(
        FObjective const& f,
        TScalar fk,
        Eigen::MatrixBase<TDerivedG> const& gk,
        Eigen::MatrixBase<TDerivedDX> const& dx,
        Eigen::MatrixBase<TDerivedX> const& xk);
};

template <class TScalar>
inline BackTrackingLineSearch<TScalar>::BackTrackingLineSearch(
    int nMaxItersIn,
    TScalar tauIn,
    TScalar cIn,
    TScalar alphaIn,
    Eigen::Index n)
    : nMaxIters(nMaxItersIn),
      tau(tauIn),
      c(cIn),
      alpha(alphaIn),
      alphaj(alphaIn),
      fj(),
      niters(0),
      xj(n)
{
}

template <class TScalar>
template <class FObjective, class TDerivedG, class TDerivedDX, class TDerivedX>
inline bool BackTrackingLineSearch<TScalar>::Solve(
    FObjective const& f,
    TScalar fk,
    Eigen::MatrixBase<TDerivedG> const& gk,
    Eigen::MatrixBase<TDerivedDX> const& dx,
    Eigen::MatrixBase<TDerivedX> const& xk)
{
    alphaj            = alpha;
    TScalar const Dfk = gk.dot(dx);
    fj                = fk;
    TScalar flinear;
    for (niters = 0; niters < nMaxIters; ++niters)
    {
        flinear = fj + (c * alphaj) * Dfk;
        xj      = xk + alphaj * dx;
        fj      = f(xj);
        if (fj <= flinear)
            break;
        alphaj *= tau;
    }
    return fj <= flinear;
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_LINESEARCH_H
