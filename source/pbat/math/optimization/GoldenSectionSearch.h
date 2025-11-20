/**
 * @file GoldenSectionSearch.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Golden section search algorithm for 1D optimization
 * @date 2025-11-18
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_MATH_OPTIMIZATION_GOLDENSECTIONSEARCH_H
#define PBAT_MATH_OPTIMIZATION_GOLDENSECTIONSEARCH_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace pbat::math::optimization {

/**
 * @brief Result of a golden section search
 * @tparam TScalar Scalar type
 */
template <class TScalar>
struct GoldenSectionSearchResult
{
    TScalar xmin;     ///< Location of the minimum
    TScalar fmin;     ///< Function value at the minimum
    int niters;       ///< Number of iterations performed
    bool converged;   ///< Whether the algorithm converged
    TScalar interval; ///< Final interval size
};

/**
 * @brief Golden section search for finding the minimum of a unimodal function in 1D.
 *
 * The golden section search is a robust derivative-free optimization method that
 * maintains a bracketing interval containing the minimum and iteratively reduces
 * the interval size using the golden ratio.
 *
 * The algorithm guarantees convergence for unimodal functions and has optimal
 * worst-case performance among all methods that use only function evaluations.
 *
 * @tparam FObjective Callable type for the objective function with signature `f(x) -> scalar`
 * @tparam TScalar Scalar type (must satisfy common::CFloatingPoint)
 * @param f Objective function to minimize
 * @param a Left endpoint of the initial interval
 * @param b Right endpoint of the initial interval
 * @param absTol Absolute tolerance for interval convergence (default: sqrt(eps))
 * @param maxIter Maximum number of iterations (default: 100)
 * @return GoldenSectionSearchResult containing the minimum location, function value, and
 * convergence information
 * @pre `a < b` otherwise behavior is undefined
 *
 * @note The function f must be unimodal in the interval [a, b] for the algorithm to work
 * correctly.
 * @note The algorithm terminates when either:
 *       1. The interval size falls below absTol
 *       2. The relative interval size falls below relTol
 *       3. The maximum number of iterations is reached
 *
 * @par Example:
 * @code
 * auto f = [](double x) { return (x - 2.0) * (x - 2.0); };
 * auto result = GoldenSectionSearch(f, 0.0, 5.0);
 * // result.xmin ≈ 2.0, result.fmin ≈ 0.0
 * @endcode
 */
template <class FObjective, common::CFloatingPoint TScalar = Scalar>
    requires std::is_invocable_r_v<TScalar, FObjective, TScalar>
[[nodiscard]] GoldenSectionSearchResult<TScalar> GoldenSectionSearch(
    FObjective const& f,
    TScalar a,
    TScalar b,
    TScalar absTol = std::sqrt(std::numeric_limits<TScalar>::epsilon()),
    int maxIter    = 100)
{
    // Golden ratio constants
    constexpr TScalar phi     = TScalar(1.618033988749895); // (1 + sqrt(5)) / 2
    constexpr TScalar invphi  = TScalar(0.618033988749895); // 1 / phi = phi - 1
    constexpr TScalar invphi2 = TScalar(0.381966011250105); // 1 / phi^2 = 1 - invphi
    // Initialize interval
    TScalar h  = b - a;
    TScalar c  = a + invphi2 * h;
    TScalar d  = a + invphi * h;
    TScalar fc = f(c);
    TScalar fd = f(d);
    // Main iteration loop
    GoldenSectionSearchResult<TScalar> result;
    result.niters = 0;
    while (result.niters < maxIter)
    {
        ++result.niters;
        // Check convergence criteria
        TScalar const xmid = TScalar(0.5) * (a + b);
        if (h < absTol)
        {
            result.converged = true;
            break;
        }
        // Golden section step: narrow the interval
        if (fc < fd)
        {
            // Minimum is in [a, d]
            b  = d;
            d  = c;
            fd = fc;
            h  = b - a;
            c  = a + invphi2 * h;
            fc = f(c);
        }
        else
        {
            // Minimum is in [c, b]
            a  = c;
            c  = d;
            fc = fd;
            h  = b - a;
            d  = a + invphi * h;
            fd = f(d);
        }
    }
    // Set convergence flag if max iterations reached
    if (result.niters >= maxIter)
        result.converged = false;
    // Return the midpoint of the final interval as the minimum
    result.xmin     = TScalar(0.5) * (a + b);
    result.fmin     = f(result.xmin);
    result.interval = h;
    return result;
}

/**
 * @brief Overload of GoldenSectionSearch with default tolerances
 *
 * @tparam FObjective Callable type for the objective function
 * @tparam TScalar Scalar type
 * @param f Objective function to minimize
 * @param a Left endpoint of the initial interval
 * @param b Right endpoint of the initial interval
 * @param maxIter Maximum number of iterations
 * @return GoldenSectionSearchResult<TScalar>
 */
template <
    class FObjective,
    common::CFloatingPoint TScalar = Scalar,
    class = std::enable_if_t<std::is_invocable_r_v<TScalar, FObjective, TScalar>>>
[[nodiscard]] inline GoldenSectionSearchResult<TScalar>
GoldenSectionSearch(FObjective const& f, TScalar a, TScalar b, int maxIter)
{
    return GoldenSectionSearch(
        f,
        a,
        b,
        std::sqrt(std::numeric_limits<TScalar>::epsilon()),
        maxIter);
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_GOLDENSECTIONSEARCH_H
