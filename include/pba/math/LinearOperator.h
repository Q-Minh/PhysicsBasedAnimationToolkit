#ifndef PBA_CORE_MATH_LINEAR_OPERATOR_H
#define PBA_CORE_MATH_LINEAR_OPERATOR_H

#include "pba/aliases.h"

#include <concepts>
#include <exception>
#include <tuple>

namespace pba {
namespace math {

/**
 * @brief Concept for operator that satisfies linearity in the mathematical sense.
 *
 * Linear operators satisfy L(ax+bz) = a*L(x) + b*L(z), hence simply scale and add the
 * input (1st parameter of Apply) prior to the Apply member function to obtain the desired result.
 * Often, the user wishes to obtain the result of multiple applications of linear operators,
 * hence we should not overwrite the out variable (2nd parameter of Apply), but simply add to it. To
 * subtract from it, simply negate the input x, i.e. L(-x) = -L(x) by linearity.
 *
 */
template <class T>
concept CLinearOperator = requires(T t)
{
    {
        t.OutputDimensions()
    } -> std::convertible_to<int>;
    {
        t.InputDimensions()
    } -> std::convertible_to<int>;
    {t.Apply(VectorX{}, std::declval<VectorX&>())};
    {t.Apply(MatrixX{}, std::declval<MatrixX&>())};
    {
        t.ToMatrix()
    } -> std::convertible_to<SparseMatrix>;
};

template <CLinearOperator... TLinearOperators>
class CompositeLinearOperator
{
  public:
    using SelfType = CompositeLinearOperator<TLinearOperators...>;

    CompositeLinearOperator(TLinearOperators const&... inOps);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Applies all linear operators on x, adding result to y.
     *
     * @tparam TDerivedIn
     * @tparam TDerivedOut
     * @param x
     * @param y
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Construct the matrix of all underlying matrices obtained by Lops.
     * @return
     */
    SparseMatrix ToMatrix() const;

    Index OutputDimensions() const;
    Index InputDimensions() const;

  private:
    std::tuple<TLinearOperators const&...> ops;
};

template <CLinearOperator... TLinearOperators>
inline CompositeLinearOperator<TLinearOperators...>::CompositeLinearOperator(
    TLinearOperators const&... inOps)
    : ops(std::make_tuple(std::cref(inOps)...))
{
    bool const bInputDimensionsMatch = std::apply(
        [this](auto... op) -> bool {
            return ((InputDimensions() == op.InputDimensions()) and ...);
        },
        ops);
    bool const bOutputDimensionsMatch = std::apply(
        [this](auto... op) -> bool {
            return ((OutputDimensions() == op.OutputDimensions()) and ...);
        },
        ops);
    if (not(bInputDimensionsMatch and bOutputDimensionsMatch))
    {
        throw std::invalid_argument(
            "Dimensionality mismatch found in CompositeLinearOperator's linear operators.");
    }
}

template <CLinearOperator... TLinearOperators>
template <class TDerivedIn, class TDerivedOut>
inline void CompositeLinearOperator<TLinearOperators...>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    std::apply([&](auto... op) { (op.Apply(x, y), ...); }, ops);
}

template <CLinearOperator... TLinearOperators>
inline SparseMatrix CompositeLinearOperator<TLinearOperators...>::ToMatrix() const
{
    SparseMatrix const M =
        std::apply([&](auto... op) -> SparseMatrix { return (op.ToMatrix() + ...); }, ops);
    return M;
}

template <CLinearOperator... TLinearOperators>
inline Index CompositeLinearOperator<TLinearOperators...>::OutputDimensions() const
{
    return std::get<0>(ops).OutputDimensions();
}

template <CLinearOperator... TLinearOperators>
inline Index CompositeLinearOperator<TLinearOperators...>::InputDimensions() const
{
    return std::get<0>(ops).InputDimensions();
}

} // namespace math
} // namespace pba

#endif // PBA_CORE_MATH_LINEAR_OPERATOR_H