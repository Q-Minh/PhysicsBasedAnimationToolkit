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
#include "pbat/common/Concepts.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/Modulo.h"
#include "pbat/io/Archive.h"

#include <cassert>
#include <exception>
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
template <class TScalar = Scalar, class TIndex = Index>
class Bdf
{
  public:
    using ScalarType = TScalar; ///< Floating point scalar type
    using IndexType  = TIndex;  ///< Integer index type

    /**
     * @brief Construct a `step`-step BDF (backward differentiation formula) time integration scheme
     * for a system of ODEs (ordinary differential equation) of order `order`.
     * @param step `0 < s < 7` backward differentiation scheme
     * @param order `order > 0` order of the ODE
     * @pre `0 < step < 7`
     * @pre `order > 0`
     */
    Bdf(int step = 1, int order = 2);

    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>
        xt; ///< `n x |s*order|` matrix of `n`-dimensional states and their time derivatives
            ///< s.t. \f$ xt.col(o*s + k) = x^(k)(t - k*dt) \f$ for \f$ k = 0, ..., s \f$ and
            ///< \f$ o = 0, ..., \text{order}-1 \f$
    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>
        xtilde;   ///< `n x order` matrix of `n`-dimensional aggregated past states and time
                  ///< derivatives s.t. xtilde.col(o) = \f$ \frac{1}{\alpha_s} \sum_{k=t_i-s}^{s-1}
                  ///< \alpha_k x_k \f$ for \f$ o = 0, ..., \text{order}-1 \f$
    IndexType ti; ///< Current time index s.t. \f$t = t_0 + h t_i\f$
    ScalarType h; ///< Time step size \f$ h \f$

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
    auto State(int k, int o = 0) const -> decltype(xt.col(0));
    /**
     * @brief \f$ o^\text{th} \f$ state derivative
     * @param k State index \f$ k = 0, ..., s \f$ for the vector \f$ x^{(o)}_{t_i - s + k} \f$
     * @param o Order of the state derivative \f$ o = 0, ..., \text{order}-1 \f$
     * @return `n x 1` state derivative vector \f$ x^{(o)}_{t_i - s + k} \f$
     */
    auto State(int k, int o = 0) -> decltype(xt.col(0));
    /**
     * @brief Current state derivative
     * @param o Order of the state derivative \f$ o = 0, ..., \text{order}-1 \f$
     * @return `n x 1` current state derivative vector \f$ x^{(o)}_{t_i} \f$
     */
    auto CurrentState(int o = 0) const -> decltype(xt.col(0));
    /**
     * @brief Current state derivative
     * @param o Order of the state derivative \f$ o = 0, ..., \text{order}-1 \f$
     * @return `n x 1` current state derivative vector \f$ x^{(o)}_{t_i} \f$
     */
    auto CurrentState(int o = 0) -> decltype(xt.col(0));
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
    void SetTimeStep(ScalarType dt);
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
    /**
     * @brief Serialize to HDF5 group
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize from HDF5 group
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive const& archive);

  private:
    int mOrder; ///< ODE order \f$ \text{order} >= 1 \f$
    int mStep;  ///< Step \f$ 0 < s < 7 \f$ backward differentiation scheme
    Eigen::Vector<ScalarType, 6>
        mAlpha; ///< Interpolation coefficients \f$ \alpha_k \f$ except \f$ \alpha_s \f$
    ScalarType
        mBeta; ///< Forcing term coefficient \f$ \beta \f$ s.t. \f$ \tilde{\beta} = h \beta \f$
};

template <class TScalar, class TIndex>
Bdf<TScalar, TIndex>::Bdf(int step, int order)
    : xt(), xtilde(), ti(0), h(Scalar(0.01)), mOrder(), mStep(), mAlpha(), mBeta()
{
    SetStep(step);
    SetOrder(order);
}

template <class TScalar, class TIndex>
auto Bdf<TScalar, TIndex>::State(int k, int o) const -> decltype(xt.col(0))
{
    if (k < 0 || k > mStep)
    {
        throw std::out_of_range("0 <= k <= s");
    }
    if (o < 0 || o >= mOrder)
    {
        throw std::out_of_range("0 <= o < order");
    }
    auto kt = common::Modulo(ti /*- mStep*/ + k, mStep);
    return xt.col(o * mStep + kt);
}

template <class TScalar, class TIndex>
auto Bdf<TScalar, TIndex>::State(int k, int o) -> decltype(xt.col(0))
{
    if (k < 0 || k > mStep)
    {
        throw std::out_of_range("0 <= k <= s");
    }
    if (o < 0 || o >= mOrder)
    {
        throw std::out_of_range("0 <= o < order");
    }
    auto kt = common::Modulo(ti /*- mStep*/ + k, mStep);
    return xt.col(o * mStep + kt);
}

template <class TScalar, class TIndex>
auto Bdf<TScalar, TIndex>::CurrentState(int o) const -> decltype(xt.col(0))
{
    return State(mStep - 1, o);
}

template <class TScalar, class TIndex>
auto Bdf<TScalar, TIndex>::CurrentState(int o) -> decltype(xt.col(0))
{
    return State(mStep - 1, o);
}

template <class TScalar, class TIndex>
void Bdf<TScalar, TIndex>::SetOrder(int order)
{
    if (order <= 0)
    {
        throw std::invalid_argument("order > 0");
    }
    mOrder = order;
}

template <class TScalar, class TIndex>
void Bdf<TScalar, TIndex>::SetStep(int step)
{
    if (step < 1 || step > 6)
    {
        throw std::invalid_argument("0 < s < 7.");
    }
    mStep = step;
    mAlpha.setZero();
    switch (mStep)
    {
        case 1:
            mAlpha(0) = Scalar(-1);
            mBeta     = Scalar(1);
            break;
        case 2:
            mAlpha(0) = Scalar(1) / 3;
            mAlpha(1) = Scalar(-4) / 3;
            mBeta     = Scalar(2) / 3;
            break;
        case 3:
            mAlpha(0) = Scalar(-2) / 11;
            mAlpha(1) = Scalar(9) / 11;
            mAlpha(2) = Scalar(-18) / 11;
            mBeta     = Scalar(6) / 11;
            break;
        case 4:
            mAlpha(0) = Scalar(3) / 25;
            mAlpha(1) = Scalar(-16) / 25;
            mAlpha(2) = Scalar(36) / 25;
            mAlpha(3) = Scalar(-48) / 25;
            mBeta     = Scalar(12) / 25;
            break;
        case 5:
            mAlpha(0) = Scalar(-12) / 137;
            mAlpha(1) = Scalar(75) / 137;
            mAlpha(2) = Scalar(-200) / 137;
            mAlpha(3) = Scalar(300) / 137;
            mAlpha(4) = Scalar(-300) / 137;
            mBeta     = Scalar(60) / 137;
            break;
        case 6:
            mAlpha(0) = Scalar(10) / 147;
            mAlpha(1) = Scalar(-72) / 147;
            mAlpha(2) = Scalar(225) / 147;
            mAlpha(3) = Scalar(-400) / 147;
            mAlpha(4) = Scalar(450) / 147;
            mAlpha(5) = Scalar(-360) / 147;
            mBeta     = Scalar(60) / 147;
            break;
    }
}

template <class TScalar, class TIndex>
void Bdf<TScalar, TIndex>::SetTimeStep(ScalarType dt)
{
    if (dt <= 0)
    {
        throw std::invalid_argument("dt > 0");
    }
    h = dt;
}

template <class TScalar, class TIndex>
void Bdf<TScalar, TIndex>::ConstructEquations()
{
    xtilde.setZero();
    for (auto o = 0; o < mOrder; ++o)
        for (auto k = 0; k < mStep; ++k)
            xtilde.col(o) += mAlpha(k) * State(k, o);
}

template <class TScalar, class TIndex>
template <class TDerivedX>
inline void Bdf<TScalar, TIndex>::SetInitialConditions(Eigen::DenseBase<TDerivedX> const& x0)
{
    ti     = 0;
    auto n = x0.rows();
    xt.resize(n, mStep * mOrder);
    xtilde.resize(n, mOrder);
    for (auto o = 0; o < mOrder; ++o)
        xt.middleCols(o * mStep, mStep).colwise() = x0.col(o);
}

template <class TScalar, class TIndex>
template <class... TDerivedX>
inline void Bdf<TScalar, TIndex>::SetInitialConditions(Eigen::DenseBase<TDerivedX> const&... x0)
{
    auto constexpr nDerivs = sizeof...(TDerivedX);
    assert(nDerivs == mOrder);
    std::tuple<decltype(x0)...> tup{x0...};
    ti     = 0;
    auto n = std::get<0>(tup).rows();
    xt.resize(n, mStep * mOrder);
    xtilde.resize(n, mOrder);
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    common::ForRange<0, nDerivs>(
        [&]<auto o>() { xt.middleCols(o * mStep, mStep).colwise() = std::get<o>(tup); });
#include "pbat/warning/Pop.h"
}

template <class TScalar, class TIndex>
template <class TDerivedX>
inline void Bdf<TScalar, TIndex>::Step(Eigen::DenseBase<TDerivedX> const& x)
{
    Tick();
    for (auto o = 0; o < mOrder; ++o)
        CurrentState(o) = x.col(o);
}

template <class TScalar, class TIndex>
template <class... TDerivedX>
inline void Bdf<TScalar, TIndex>::Step(Eigen::DenseBase<TDerivedX> const&... x)
{
    auto constexpr nDerivs = sizeof...(TDerivedX);
    assert(nDerivs == mOrder);
    Tick();
    std::tuple<decltype(x)...> tup{x...};
    common::ForRange<0, nDerivs>([&]<auto o>() { CurrentState(o) = std::get<o>(tup); });
}

template <class TScalar, class TIndex>
void Bdf<TScalar, TIndex>::Serialize(io::Archive& archive) const
{
    io::Archive bdfArchive = archive["pbat.sim.integration.Bdf"];
    bdfArchive.WriteData("xt", xt);
    bdfArchive.WriteData("xtilde", xtilde);
    bdfArchive.WriteMetaData("h", h);
    bdfArchive.WriteMetaData("ti", ti);
    bdfArchive.WriteMetaData("order", mOrder);
    bdfArchive.WriteMetaData("step", mStep);
    bdfArchive.WriteMetaData("alpha", mAlpha);
    bdfArchive.WriteMetaData("beta", mBeta);
}

template <class TScalar, class TIndex>
void Bdf<TScalar, TIndex>::Deserialize(io::Archive const& archive)
{
    io::Archive const group = archive["pbat.sim.integration.Bdf"];
    xt                      = group.ReadData<MatrixX>("xt");
    xtilde                  = group.ReadData<MatrixX>("xtilde");
    h                       = group.ReadMetaData<Scalar>("h");
    ti                      = group.ReadMetaData<Index>("ti");
    mOrder                  = group.ReadMetaData<int>("order");
    mStep                   = group.ReadMetaData<int>("step");
    mAlpha                  = group.ReadMetaData<Vector<6>>("alpha");
    mBeta                   = group.ReadMetaData<Scalar>("beta");
}

} // namespace pbat::sim::integration

#endif // PBAT_SIM_INTEGRATION_BDF_H
