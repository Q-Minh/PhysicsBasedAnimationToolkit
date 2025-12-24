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
#include "pbat/io/Archive.h"

#include <Eigen/Core>
#include <optional>
#include <type_traits>
#include <variant>

namespace pbat::math::optimization {

/**
 * @brief Newton's method for optimization
 * @tparam TScalar Scalar type
 *
 * @pre The Hessian is assumed to be positive definite at each iteration.
 * @note There are 2 code paths for objective function evaluation in our Newton's method
 * API for efficiency reasons. Often times, operations required for evaluating
 * derivatives (gradient, hessian) overlap with those required for evaluating the objective
 * function itself. To avoid redundant computations, we provide this `fPrepareDerivatives`
 * callback that is called once per iteration to compute any shared quantities. The callback
 * `FObjective` found in the `Iterate` and `Solve` methods is the second code path for objective
 * function evaluation, which is used during line search, and may be called many times per
 * iteration. The user should (probably) implement each code path specifically for their associated
 * use case. If no line search is specified, then we don't need to compute the objective function
 * value at all, but since operations for computing the derivatives already compute the objective
 * function value (indirectly), there is no redundant computation. Also, the return value of
 * `fPrepareDerivatives` is not used in that scenario.
 * @note We expose both the high-level `Solve` method and the lower-level `Iterate` and
 * `PrepareNextIteration` methods to give users more flexibility in how they want to use the Newton
 * optimizer. A custom termination criterion can be implemented by calling `PrepareNextIteration`
 * and `Iterate` in a loop manually.
 */
template <class TScalar = Scalar>
struct Newton
{
    using LineSearchType =
        std::variant<std::monostate, BackTrackingLineSearch<TScalar>>; ///< Line search type

    int nMaxIters; ///< Maximum number of iterations for the Newton solver
    TScalar gtol2; ///< Gradient squared norm threshold for convergence
    Eigen::Vector<TScalar, Eigen::Dynamic> dxk; ///< Step direction
    Eigen::Vector<TScalar, Eigen::Dynamic> gk;  ///< Gradient at current iteration
    LineSearchType lineSearch;                  ///< Line search object

    TScalar fk;      ///< Objective function value at current iteration
    TScalar gknorm2; ///< Squared norm of the gradient at current iteration
    int k;           ///< Current iteration

    /**
     * @brief Construct a new Newton optimizer
     * @param nMaxIters Maximum number of iterations for the Newton solver
     * @param gtol Gradient norm threshold for convergence
     * @param n Number of degrees of freedom
     * @param lineSearchIn Optional line search object
     */
    Newton(
        int nMaxIters               = 10,
        TScalar gtol                = TScalar(1e-4),
        Index n                     = 0,
        LineSearchType lineSearchIn = {});
    /**
     * @brief Resets the iteration index to 0 and allocates (if necessary) for the gradient.
     *
     * @tparam TDerivedX Derived type for the input iterate
     * @param xk Current iterate
     * @post `k == 0`
     */
    template <class TDerivedX>
    void InitializeSolve(Eigen::MatrixBase<TDerivedX> const& xk);
    /**
     * @brief Calls fPrepareDerivatives and evaluates the objective function and gradient at `xk`
     *
     * @tparam FPrepareDerivatives Callable type with signature
     * `fPrepareDerivatives(xk) -> TScalar`
     * @tparam FGradient Callable type with signature `g(xk, gk) -> void` that computes the gradient
     * at `xk` and stores it in `gk`
     * @tparam TDerivedX Derived type for the input iterate
     * @param fPrepareDerivatives Callback to compute any quantities necessary prior to evaluating
     * the objective function gradient and hessian. It must also return the objective function value
     * at `xk`.
     * @param g Gradient function
     * @param xk Current iterate
     */
    template <class FPrepareDerivatives, class FGradient, class TDerivedX>
    void PrepareNextIteration(
        FPrepareDerivatives const& fPrepareDerivatives,
        FGradient const& g,
        Eigen::MatrixBase<TDerivedX> const& xk);
    /**
     * @brief Perform a single Newton iteration
     *
     * The objective function is not called if no line search is specified, otherwise it
     * is called at least once.
     *
     * @tparam FObjective Callable type for the objective function with signature `f(xk) -> TScalar`
     * @tparam FHessianInverseProduct Callable type for the Hessian inverse product with signature
     * `Hinv(xk, ngk, dxk) -> void` which computes the product of the inverse Hessian at `xk` with
     * the negative gradient `ngk` and stores the result in `dxk`.
     * @tparam TDerivedX Derived type for the input iterate
     * @param f Objective function
     * @param Hinv Hessian inverse product function
     * @param xk Current iterate
     * @return true if step was taken, false otherwise
     */
    template <class FObjective, class FHessianInverseProduct, class TDerivedX>
    bool Iterate(
        FObjective const& f,
        FHessianInverseProduct const& Hinv,
        Eigen::MatrixBase<TDerivedX>& xk);
    /**
     * @brief Solve the optimization problem using Newton's method
     *
     * @tparam FPrepareDerivatives Callable type with signature
     * `fPrepareDerivatives(xk) -> void`
     * @tparam FObjective Callable type for the objective function with signature `f(xk) -> fk`
     * @tparam FGradient Callable type for the gradient with signature `g(xk, gk) -> void` that
     * computes the gradient at `xk` and stores it in `gk`
     * @tparam FHessianInverseProduct Callable type for the Hessian inverse product with signature
     * `Hinv(xk, gk, dxk) -> void` which computes the product of the inverse Hessian at `xk` with
     * the gradient `gk` and stores the result in `dxk`.
     * @tparam TDerivedX Derived type for the input iterate
     * @param fPrepareDerivatives Derivative (pre)computation function
     * @param f Objective function
     * @param g Gradient function
     * @param Hinv Hessian inverse product function
     * @param xk Current iterate
     * @param lineSearch Optional line search object
     * @return true if converged, false otherwise
     */
    template <
        class FPrepareDerivatives,
        class FObjective,
        class FGradient,
        class FHessianInverseProduct,
        class TDerivedX>
    bool Solve(
        FPrepareDerivatives const& fPrepareDerivatives,
        FObjective f,
        FGradient g,
        FHessianInverseProduct Hinv,
        Eigen::MatrixBase<TDerivedX>& xk);
    /**
     * @brief Serialize this
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize this
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive& archive);
};

template <class TScalar>
inline Newton<TScalar>::Newton(int nMaxItersIn, TScalar gtol, Index n, LineSearchType lineSearchIn)
    : nMaxIters(nMaxItersIn),
      gtol2(gtol * gtol),
      dxk(n),
      gk(n),
      lineSearch(std::move(lineSearchIn)),
      gknorm2(),
      fk()
{
}

template <class TScalar>
template <class TDerivedX>
inline void Newton<TScalar>::InitializeSolve(Eigen::MatrixBase<TDerivedX> const& xk)
{
    k = 0;
    gk.resize(xk.size());
}

template <class TScalar>
template <class FPrepareDerivatives, class FGradient, class TDerivedX>
inline void Newton<TScalar>::PrepareNextIteration(
    FPrepareDerivatives const& fPrepareDerivatives,
    FGradient const& g,
    Eigen::MatrixBase<TDerivedX> const& xk)
{
    fk = fPrepareDerivatives(xk);
    g(xk, gk);
    gknorm2 = gk.squaredNorm();
}

template <class TScalar>
template <class FObjective, class FHessianInverseProduct, class TDerivedX>
inline bool Newton<TScalar>::Iterate(
    FObjective const& f,
    FHessianInverseProduct const& Hinv,
    Eigen::MatrixBase<TDerivedX>& xk)
{
    Hinv(xk, gk, dxk);
    bool bStepped{false};
    std::visit(
        [&](auto&& lineSearch) {
            bool constexpr bNoLineSearch =
                std::is_same_v<std::decay_t<decltype(lineSearch)>, std::monostate>;
            if constexpr (bNoLineSearch)
            {
                xk -= dxk;
                bStepped = true;
            }
            else
            {
                dxk      = -dxk;
                bStepped = lineSearch.Solve(f, fk, gk, dxk, xk);
                if (bStepped)
                    xk += lineSearch.alphaj * dxk;
            }
        },
        lineSearch);
    ++k;
    return bStepped;
}

template <class TScalar>
template <
    class FPrepareDerivatives,
    class FObjective,
    class FGradient,
    class FHessianInverseProduct,
    class TDerivedX>
inline bool Newton<TScalar>::Solve(
    FPrepareDerivatives const& fPrepareDerivatives,
    FObjective f,
    FGradient g,
    FHessianInverseProduct Hinv,
    Eigen::MatrixBase<TDerivedX>& xk)
{
    PrepareNextIteration(fPrepareDerivatives, g, xk.derived());
    for (; k < nMaxIters;)
    {
        if (gknorm2 < gtol2)
            return true;
        // If a step could not be taken, further Newton iterations will similarly not yield any
        // step, since both the gradient and Hessian will remain the same. We can thus terminate
        // early without convergence.
        if (not Iterate(f, Hinv, xk))
            return false;
        PrepareNextIteration(fPrepareDerivatives, g, xk);
    }
    return gknorm2 < gtol2;
}

template <class TScalar>
inline void Newton<TScalar>::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.math.optimization.Newton"];
    group.WriteMetaData("nMaxIters", nMaxIters);
    group.WriteMetaData("gtol2", gtol2);
    group.WriteData("dxk", dxk);
    group.WriteData("gk", gk);
    std::visit(
        [&](auto&& lineSearch) {
            using U = std::decay_t<decltype(lineSearch)>;
            if constexpr (not std::is_same_v<U, std::monostate>)
            {
                lineSearch.Serialize(group);
            }
        },
        lineSearch);
    group.WriteMetaData("fk", fk);
    group.WriteMetaData("gknorm2", gknorm2);
    group.WriteMetaData("k", k);
}

template <class TScalar>
inline void Newton<TScalar>::Deserialize(io::Archive& archive)
{
    io::Archive group = archive["pbat.math.optimization.Newton"];
    nMaxIters         = group.ReadMetaData<std::decay_t<decltype(nMaxIters)>>("nMaxIters");
    gtol2             = group.ReadMetaData<std::decay_t<decltype(gtol2)>>("gtol2");
    dxk               = group.ReadData<std::decay_t<decltype(dxk)>>("dxk");
    gk                = group.ReadData<std::decay_t<decltype(gk)>>("gk");
    std::visit(
        [&](auto&& lineSearch) {
            using U = std::decay_t<decltype(lineSearch)>;
            if constexpr (not std::is_same_v<U, std::monostate>)
            {
                lineSearch.Deserialize(group);
            }
        },
        lineSearch);
    fk      = group.ReadMetaData<std::decay_t<decltype(fk)>>("fk");
    gknorm2 = group.ReadMetaData<std::decay_t<decltype(gknorm2)>>("gknorm2");
    k       = group.ReadMetaData<std::decay_t<decltype(k)>>("k");
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_NEWTON_H
