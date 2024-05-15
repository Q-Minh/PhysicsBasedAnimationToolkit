#ifndef PBA_CORE_MATH_LINEAR_OPERATOR_H
#define PBA_CORE_MATH_LINEAR_OPERATOR_H

#include "pba/aliases.h"

#include <concepts>

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
    {t.Apply(VectorX{}, std::declval<VectorX&>())};
    {t.Apply(MatrixX{}, std::declval<MatrixX&>())};
    {
        t.ToMatrix()
    } -> std::convertible_to<SparseMatrix>;
};

} // namespace math
} // namespace pba

#endif // PBA_CORE_MATH_LINEAR_OPERATOR_H