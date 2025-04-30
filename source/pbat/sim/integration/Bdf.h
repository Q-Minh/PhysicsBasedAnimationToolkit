/**
 * @file Bdf.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief BDF (Backward Differentiation Formula) time integration scheme
 * @date 2025-04-29
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_INTEGRATION_BDF_H
#define PBAT_SIM_INTEGRATION_BDF_H

#include "pbat/Aliases.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/Modulo.h"

#include <cassert>
#include <tuple>

namespace pbat::sim::integration {

/**
 * @brief BDF (Backward Differentiation Formula) time integration scheme for a system of ODEs for an
 * initial value problem (IVP)
 *
 * Refer to background [here](https://en.wikipedia.org/wiki/Backward_differentiation_formula).
 *
 * An order \f$ p \f$ system of ODEs \f$ x^{(p)} = f(t, \frac{d^{p-1}}{dt^{p-1}} x, \dots, x) \f$
 * can be transformed into a system of \f$ p \f$ first-order ODEs using slack variables \f$ x^{(o)}
 * = \frac{d^o}{dt^o} x \f$ for \f$ o = 0, \dots, p-1 \f$, such that the IVP can be rewritten as
 * \f[
 * \begin{align*}
 * \frac{d}{dt} x_{p-1} &= f(t, x^{(p-1)}, \dots, x^{(o)}) \\
 * \frac{d}{dt} x^{(p-j)} &= x^{(p-j+1)}, \quad j = 2, \dots, p
 * \end{align*} .
 * \f]
 *
 * Since BDF discretizes each equation as \f$ \sum_{k=0}^s \alpha_k x_{n-s+k} = h \beta f(t,
 * x_n^{(p-1)}, \dots, x_n) \f$, we have that
 * \f[
 * \begin{align*}
 * x^{(p-1)}_n + \tilde{x}^{(p-1)}_\text{BDFS} - \tilde{\beta}_\text{BDFS} f(t, x_n^{(p-1)}, \dots,
 * x_n) &= 0 \\ x^{(o)}_n + \tilde{x}^{(o)}_\text{BDFS} - \tilde{\beta}_\text{BDFS} x^{(o+1)}_n &= 0
 * \end{align*}
 * \f]
 * for \f$ o=0,\dots,p-2 \f$, where \f$ \tilde{\beta} = h \beta \f$ and \f$ n \f$ is the next time
 * step index.
 *
 * One performs time integration by solving for the states and their derivatives \f$ x_n^{(o)} \f$
 * for
 * \f$ o=0,\dots,p-1 \f$ at each time step using root-finding of the above equations. Alternatively,
 * the root-finding equations can be treated as stationarity conditions \f$ \nabla f = 0 \f$ for
 * some objective function \f$ f \f$, and solved via numerical optimization of \f$ f \f$.
 *
 * This class encapsulates the storage of the past states and their lower order derivatives \f$
 * x_{n-s+k}^{(o)} \f$, the construction of so-called inertias \f$ \tilde{x}^{(o)}_\text{BDFS} \f$,
 * the interpolation coefficients \f$ \alpha_k \f$, forcing term coefficient \f$ \beta \f$ and
 * generalize the BDF scheme to various ODE systems of different orders. Specific ODEs are then
 * entirely defined by the forcing function \f$ f(t, \frac{d^{p-1}}{dt^{p-1}} x, \dots, x) \f$, and
 * it is up to the user to derive their specific equations to solve.
 *
 * ```
 * bdf.SetInitialConditions(x0, v0)
 * for ti in steps:
 *   bdf.ConstructEquations()
 *   xn, vn = userSolve(bdf)
 *   bdf.Step(xn, vn)
 * ```
 *
 */
class Bdf
{
  public:
    /**
     * @brief Construct a `step`-step BDF (backward differentiation formula) time integration scheme
     * for a system of ODEs (ordinary differential equation) of order `order`.
     * @param step `0 < s < 7` backward differentiation scheme
     * @param order `order > 0` order of the ODE
     * @pre `0 < step < 7`
     * @pre `order > 0`
     */
    Bdf(int step = 1, int order = 2);

    MatrixX xt; ///< `n x |s*order|` matrix of `n`-dimensional states and their time derivatives
                ///< s.t. \f$ xt.col(o*s + k) = x^(k)(t - k*dt) \f$ for \f$ k = 0, ..., s \f$ and
                ///< \f$ o = 0, ..., \text{order}-1 \f$
    MatrixX
        xtilde; ///< `n x order` matrix of `n`-dimensional aggregated past states and time
                ///< derivatives s.t. xtilde.col(o) = \f$ \frac{1}{\alpha_s} \sum_{k=t_i-s}^{s-1}
                ///< \alpha_k x_k \f$ for \f$ o = 0, ..., \text{order}-1 \f$
    Index ti;   ///< Current time index s.t. \f$t = t_0 + h t_i\f$
    Scalar h;   ///< Time step size \f$ h \f$

    /**
     * @brief Order of the ODE
     * @return Order of the ODE
     */
    [[maybe_unused]] auto Order() const { return mOrder; }
    /**
     * @brief Step `s` of the `s`-step BDF scheme
     * @return Step `s` of the `s`-step BDF scheme
     */
    [[maybe_unused]] auto Step() const { return mStep; }
    /**
     * @brief Number of ODEs
     * @return Number of ODEs
     */
    [[maybe_unused]] auto Dimensions() const { return xt.rows(); }
    /**
     * @brief Time step size
     * @return Time step size
     */
    [[maybe_unused]] auto TimeStep() const { return h; }
    /**
     * @brief Inertia of the BDF scheme for the \f$ o^\text{th} \f$ state derivative
     * @param o Order of the state derivative \f$ o = 0, ..., \text{order}-1 \f$
     * @return `n x 1` inertia vector for the \f$ o^\text{th} \f$ state derivative
     */
    [[maybe_unused]] auto Inertia(int o) const { return xtilde.col(o); }
    /**
     * @brief \f$ o^\text{th} \f$ state derivative
     * @param k State index \f$ k = 0, ..., s \f$ for the vector \f$ x^{(o)}_{t_i - s + k} \f$
     * @param o Order of the state derivative \f$ o = 0, ..., \text{order}-1 \f$
     * @return `n x 1` state derivative vector \f$ x^{(o)}_{t_i - s + k} \f$
     */
    auto State(int k, int o = 0) const;
    /**
     * @brief \f$ o^\text{th} \f$ state derivative
     * @param k State index \f$ k = 0, ..., s \f$ for the vector \f$ x^{(o)}_{t_i - s + k} \f$
     * @param o Order of the state derivative \f$ o = 0, ..., \text{order}-1 \f$
     * @return `n x 1` state derivative vector \f$ x^{(o)}_{t_i - s + k} \f$
     */
    auto State(int k, int o = 0);
    /**
     * @brief Current state derivative
     * @param o Order of the state derivative \f$ o = 0, ..., \text{order}-1 \f$
     * @return `n x 1` current state derivative vector \f$ x^{(o)}_{t_i} \f$
     */
    auto CurrentState(int o = 0) const;
    /**
     * @brief Current state derivative
     * @param o Order of the state derivative \f$ o = 0, ..., \text{order}-1 \f$
     * @return `n x 1` current state derivative vector \f$ x^{(o)}_{t_i} \f$
     */
    auto CurrentState(int o = 0);
    /**
     * @brief Interpolation coefficients \f$ \alpha_k \f$ except \f$ \alpha_s \f$
     * @return `s x 1` vector of interpolation coefficients \f$ \alpha_k \f$
     */
    [[maybe_unused]] auto Alpha() const { return mAlpha(Eigen::seqN(0, mStep)); }
    /**
     * @brief Forcing term coefficient \f$ \beta \f$ s.t. \f$ \tilde{\beta} = h \beta \f$
     * @return Forcing term coefficient \f$ \beta \f$
     */
    [[maybe_unused]] auto Beta() const { return mBeta; }
    /**
     * @brief Time-step scaled forcing term coefficient \f$ \tilde{\beta} = h \beta \f$
     * @return Time-step scaled forcing term coefficient \f$ \tilde{\beta} = h \beta \f$
     */
    [[maybe_unused]] auto BetaTilde() const { return mBeta * h; }

    /**
     * @brief Set the order of the ODE system
     * @param order Order of the ODE system \f$ \text{order} > 0 \f$
     * @pre `order > 0`
     */
    void SetOrder(int order);
    /**
     * @brief Set the step of the BDF scheme
     * @param step Step of the BDF scheme \f$ 0 < s < 7 \f$
     * @pre `0 < step < 7`
     */
    void SetStep(int step);
    /**
     * @brief Set the time step size
     * @param dt Time step size \f$ dt > 0 \f$
     * @pre `dt > 0`
     */
    void SetTimeStep(Scalar dt);
    /**
     * @brief Construct the BDF equations, i.e. compute \f$ \tilde{x^{(o)}} = \sum_{k=0}^{s-1}
     * \alpha_k x^{(o)}_{t_i - s + k} \f$ for all \f$ o = 0, ..., \text{order}-1 \f$
     */
    void ConstructEquations();
    /**
     * @brief Set the initial conditions for the initial value problem
     * @param x0 `n x order` matrix of initial conditions s.t. `x0.col(o) = \f$ x^{(o)}_{t_0}
     * \f$` for \f$ o = 0, ..., \text{order}-1 \f$
     * @pre `x0.cols() == order`
     * @post `ti == 0`
     */
    template <class TDerivedX>
    void SetInitialConditions(Eigen::DenseBase<TDerivedX> const& x0);
    /**
     * @brief Set the initial conditions for the initial value problem
     * @param x0 `order` vectors of `n x 1` initial conditions \f$ x^{(o)}_{t_0} \f$
     * @pre `sizeof...(TDerivedX) == order`
     * @post `ti == 0`
     */
    template <class... TDerivedX>
    void SetInitialConditions(Eigen::DenseBase<TDerivedX> const&... x0);
    /**
     * @brief Advance the BDF scheme by one time step
     * @tparam TDerivedX Derived type of the input matrix
     * @param x `n x order` matrix of the current state derivatives \f$ x_{t_i}^{(o)} \f$
     */
    template <class TDerivedX>
    void Step(Eigen::DenseBase<TDerivedX> const& x);
    /**
     * @brief Advance the BDF scheme by one time step
     * @tparam TDerivedX Derived type of the input matrix
     * @param xs `order` vectors of `n x 1` current state derivatives \f$ x_{t_i}^{(o)} \f$
     * @pre `sizeof...(TDerivedX) == order`
     */
    template <class... TDerivedX>
    void Step(Eigen::DenseBase<TDerivedX> const&... xs);
    /**
     * @brief Advance the BDF scheme by one time step
     */
    [[maybe_unused]] void Tick() { ++ti; }

  private:
    int mOrder;       ///< ODE order \f$ \text{order} >= 1 \f$
    int mStep;        ///< Step \f$ 0 < s < 7 \f$ backward differentiation scheme
    Vector<6> mAlpha; ///< Interpolation coefficients \f$ \alpha_k \f$ except \f$ \alpha_s \f$
    Scalar mBeta; ///< Forcing term coefficient \f$ \beta \f$ s.t. \f$ \tilde{\beta} = h \beta \f$
};

template <class TDerivedX>
inline void Bdf::SetInitialConditions(Eigen::DenseBase<TDerivedX> const& x0)
{
    ti     = 0;
    auto n = x0.rows();
    xt.resize(n, mStep * mOrder);
    xtilde.resize(n, mOrder);
    for (auto o = 0; o < mOrder; ++o)
        xt.middleCols(o * mStep, mStep).colwise() = x0.col(o);
}

template <class... TDerivedX>
inline void Bdf::SetInitialConditions(Eigen::DenseBase<TDerivedX> const&... x0)
{
    auto constexpr nDerivs = sizeof...(TDerivedX);
    assert(nDerivs == mOrder);
    ti     = 0;
    auto n = x0.rows();
    xt.resize(n, mStep * mOrder);
    xtilde.resize(n, mOrder);
    std::tuple<decltype(x0)...> tup{x0...};
    common::ForRange<0, nDerivs>(
        [&]<auto o>() { xt.middleCols(o * mStep, mStep).colwise() = std::get<o>(tup); });
}

template <class TDerivedX>
inline void Bdf::Step(Eigen::DenseBase<TDerivedX> const& x)
{
    Tick();
    for (auto o = 0; o < mOrder; ++o)
        CurrentState(o) = x.col(o);
}

template <class... TDerivedX>
inline void Bdf::Step(Eigen::DenseBase<TDerivedX> const&... x)
{
    auto constexpr nDerivs = sizeof...(TDerivedX);
    assert(nDerivs == mOrder);
    Tick();
    std::tuple<decltype(x)...> tup{x...};
    common::ForRange<0, nDerivs>([&]<auto o>() { CurrentState(o) = std::get<o>(tup); });
}

} // namespace pbat::sim::integration

#endif // PBAT_SIM_INTEGRATION_BDF_H
