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
#include "pbat/io/Archive.h"

#include <Eigen/Core>
#include <type_traits>

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
    TScalar flinearj;                          ///< Current linearized objective function value
    TScalar Dfk;                               ///< Directional derivative at current step
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

    /**
     * @brief Initialize the line search state for a new search
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
     */
    template <class FObjective, class TDerivedG, class TDerivedDX, class TDerivedX>
    void Initialize(
        FObjective const& f,
        TScalar fk,
        Eigen::MatrixBase<TDerivedG> const& gk,
        Eigen::MatrixBase<TDerivedDX> const& dx,
        Eigen::MatrixBase<TDerivedX> const& xk);

    /**
     * @brief Perform one iteration of the backtracking update
     *
     * Uses the stored directional derivative and current step size to update
     * the candidate iterate and its objective value.
     *
     * @tparam FObjective Callable type for the objective function with signature `f(x) -> f(x)`
     * @tparam TDerivedDX Derived type for the step direction
     * @tparam TDerivedX Derived type for the current iterate
     * @param f Objective function
     * @param fk Objective function value at the initial iterate
     * @param dx Step direction
     * @param xk Initial iterate
     */
    template <class FObjective, class TDerivedDX, class TDerivedX>
    void Iterate(
        FObjective const& f,
        TScalar fk,
        Eigen::MatrixBase<TDerivedDX> const& dx,
        Eigen::MatrixBase<TDerivedX> const& xk);
    /**
     * @brief Serialize the line search parameters and state
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the line search parameters and state
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive& archive);
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
inline void BackTrackingLineSearch<TScalar>::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.math.optimization.BackTrackingLineSearch"];
    group.WriteMetaData("nMaxIters", nMaxIters);
    group.WriteMetaData("tau", tau);
    group.WriteMetaData("c", c);
    group.WriteMetaData("alpha", alpha);
    group.WriteMetaData("alphaj", alphaj);
    group.WriteMetaData("fj", fj);
    group.WriteMetaData("flinearj", flinearj);
    group.WriteMetaData("Dfk", Dfk);
    group.WriteData("xj", xj);
    group.WriteMetaData("niters", niters);
}

template <class TScalar>
inline void BackTrackingLineSearch<TScalar>::Deserialize(io::Archive& archive)
{
    io::Archive group = archive["pbat.math.optimization.BackTrackingLineSearch"];
    nMaxIters         = group.ReadMetaData<std::decay_t<decltype(nMaxIters)>>("nMaxIters");
    tau               = group.ReadMetaData<std::decay_t<decltype(tau)>>("tau");
    c                 = group.ReadMetaData<std::decay_t<decltype(c)>>("c");
    alpha             = group.ReadMetaData<std::decay_t<decltype(alpha)>>("alpha");
    alphaj            = group.ReadMetaData<std::decay_t<decltype(alphaj)>>("alphaj");
    fj                = group.ReadMetaData<std::decay_t<decltype(fj)>>("fj");
    flinearj          = group.ReadMetaData<std::decay_t<decltype(flinearj)>>("flinearj");
    Dfk               = group.ReadMetaData<std::decay_t<decltype(Dfk)>>("Dfk");
    xj                = group.ReadData<std::decay_t<decltype(xj)>>("xj");
    niters            = group.ReadMetaData<std::decay_t<decltype(niters)>>("niters");
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
    Initialize(f, fk, gk, dx, xk);
    while (niters < nMaxIters)
    {
        if (fj <= flinearj)
            break;
        Iterate(f, fk, dx, xk);
    }
    return fj <= flinearj;
}

template <class TScalar>
template <class FObjective, class TDerivedG, class TDerivedDX, class TDerivedX>
inline void BackTrackingLineSearch<TScalar>::Initialize(
    FObjective const& f,
    TScalar fk,
    Eigen::MatrixBase<TDerivedG> const& gk,
    Eigen::MatrixBase<TDerivedDX> const& dx,
    Eigen::MatrixBase<TDerivedX> const& xk)
{
    niters   = 0;
    alphaj   = alpha;
    Dfk      = gk.dot(dx);
    flinearj = fk + (c * alphaj) * Dfk;
    xj       = xk + alphaj * dx;
    fj       = f(xj);
}

template <class TScalar>
template <class FObjective, class TDerivedDX, class TDerivedX>
inline void BackTrackingLineSearch<TScalar>::Iterate(
    FObjective const& f,
    TScalar fk,
    Eigen::MatrixBase<TDerivedDX> const& dx,
    Eigen::MatrixBase<TDerivedX> const& xk)
{
    alphaj *= tau;
    flinearj = fk + (c * alphaj) * Dfk;
    xj       = xk + alphaj * dx;
    fj       = f(xj);
    ++niters;
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_LINESEARCH_H
