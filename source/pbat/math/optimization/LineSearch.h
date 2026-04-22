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
    TScalar mj;                                ///< Current merit function value
    TScalar mlinearj;                          ///< Current linearized merit function value
    TScalar Dm0;                               ///< Directional derivative of merit at alpha=0
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
     * @brief Perform a backtracking line search using a general merit function
     *
     * The Armijo condition is: merit(xk + alpha*dx) <= mk + c*alpha*Dm0
     *
     * @tparam FMerit Callable type for the merit function with signature `merit(x) -> TScalar`
     * @tparam TDerivedDX Derived type for the step direction
     * @tparam TDerivedX Derived type for the current iterate
     * @param merit Merit function (e.g., objective, or objective + penalty)
     * @param mk Merit function value at the current iterate
     * @param Dm0In Directional derivative of merit at xk in direction dx
     * @param dx Step direction
     * @param xk Current iterate
     * @return true if the line search succeeded, false otherwise
     */
    template <class FMerit, class TDerivedDX, class TDerivedX>
    bool Solve(
        FMerit const& merit,
        TScalar mk,
        TScalar Dm0In,
        Eigen::MatrixBase<TDerivedDX> const& dx,
        Eigen::MatrixBase<TDerivedX> const& xk);
    /**
     * @brief Serialize the line search parameters and state
     * @param archive Archive to serialize to
     * @param bMinimal If true, only serialize essential data
     */
    void Serialize(io::Archive& archive, bool bMinimal = true) const;
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
      mj(),
      niters(0),
      xj(n)
{
}

template <class TScalar>
template <class FMerit, class TDerivedDX, class TDerivedX>
inline bool BackTrackingLineSearch<TScalar>::Solve(
    FMerit const& merit,
    TScalar mk,
    TScalar Dm0In,
    Eigen::MatrixBase<TDerivedDX> const& dx,
    Eigen::MatrixBase<TDerivedX> const& xk)
{
    alphaj   = alpha;
    Dm0      = Dm0In;
    mlinearj = mk + (c * alphaj) * Dm0;
    xj       = xk + alphaj * dx;
    mj       = merit(xj);
    for (niters = 0; niters < nMaxIters; ++niters)
    {
        if (mj <= mlinearj)
            break;
        alphaj *= tau;
        mlinearj = mk + (c * alphaj) * Dm0;
        xj       = xk + alphaj * dx;
        mj       = merit(xj);
    }
    return mj <= mlinearj;
}

template <class TScalar>
inline void BackTrackingLineSearch<TScalar>::Serialize(io::Archive& archive, bool bMinimal) const
{
    io::Archive group = archive["pbat.math.optimization.BackTrackingLineSearch"];
    group.WriteMetaData("nMaxIters", nMaxIters);
    group.WriteMetaData("tau", tau);
    group.WriteMetaData("c", c);
    group.WriteMetaData("alpha", alpha);
    group.WriteMetaData("alphaj", alphaj);
    group.WriteMetaData("mj", mj);
    group.WriteMetaData("mlinearj", mlinearj);
    group.WriteMetaData("Dm0", Dm0);
    if (not bMinimal)
    {
        group.WriteData("xj", xj);
    }
    group.WriteMetaData("niters", niters);
}

template <class TScalar>
inline void BackTrackingLineSearch<TScalar>::Deserialize(io::Archive& archive)
{
    io::Archive group = archive["pbat.math.optimization.BackTrackingLineSearch"];
    if (group.HasMetaData("nMaxIters"))
        nMaxIters = group.ReadMetaData<std::decay_t<decltype(nMaxIters)>>("nMaxIters");
    if (group.HasMetaData("tau"))
        tau = group.ReadMetaData<std::decay_t<decltype(tau)>>("tau");
    if (group.HasMetaData("c"))
        c = group.ReadMetaData<std::decay_t<decltype(c)>>("c");
    if (group.HasMetaData("alpha"))
        alpha = group.ReadMetaData<std::decay_t<decltype(alpha)>>("alpha");
    if (group.HasMetaData("alphaj"))
        alphaj = group.ReadMetaData<std::decay_t<decltype(alphaj)>>("alphaj");
    if (group.HasMetaData("mj"))
        mj = group.ReadMetaData<std::decay_t<decltype(mj)>>("mj");
    if (group.HasMetaData("mlinearj"))
        mlinearj = group.ReadMetaData<std::decay_t<decltype(mlinearj)>>("mlinearj");
    if (group.HasMetaData("Dm0"))
        Dm0 = group.ReadMetaData<std::decay_t<decltype(Dm0)>>("Dm0");
    if (group.HasData("xj"))
        xj = group.ReadData<std::decay_t<decltype(xj)>>("xj");
    if (group.HasMetaData("niters"))
        niters = group.ReadMetaData<std::decay_t<decltype(niters)>>("niters");
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_LINESEARCH_H
