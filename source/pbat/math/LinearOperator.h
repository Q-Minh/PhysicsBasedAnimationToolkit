/**
 * @file LinearOperator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Linear operator concept and composite type
 * @date 2025-02-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PBAT_MATH_LINEAR_OPERATOR_H
#define PBAT_MATH_LINEAR_OPERATOR_H

#include <concepts>
#include <exception>
#include <pbat/Aliases.h>
#include <pbat/profiling/Profiling.h>
#include <tuple>

namespace pbat {
namespace math {

/**
 * @brief Concept for operator that satisfies linearity in the mathematical sense.
 *
 * Linear operators satisfy \f$ L(ax+bz) = a*L(x) + b*L(z) \f$, hence simply scale and add the
 * input (1st parameter of Apply) prior to the Apply member function to obtain the desired result.
 * Often, the user wishes to obtain the result of multiple applications of linear operators,
 * hence we should not overwrite the out variable (2nd parameter of Apply), but simply add to it. To
 * subtract from it, simply negate the input x, i.e. \f$ L(-x) = -L(x) \f$ by linearity.
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
    } -> std::convertible_to<CSCMatrix>;
};

template <CLinearOperator... TLinearOperators>
class LinearOperator;

} // namespace math
} // namespace pbat

namespace Eigen {
namespace internal {

template <pbat::math::CLinearOperator... TLinearOperators>
struct traits<pbat::math::LinearOperator<TLinearOperators...>>
{
    using Scalar       = pbat::Scalar;
    using StorageIndex = pbat::CSCMatrix::StorageIndex;
    using StorageKind  = Sparse;
    using XprKind      = MatrixXpr;
    enum {
        RowsAtCompileTime    = Dynamic,
        ColsAtCompileTime    = Dynamic,
        MaxRowsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic,
        Flags                = 0
    };
};

} // namespace internal
} // namespace Eigen

namespace pbat {
namespace math {

/**
 * @brief Zero-overhead composite type satisfying the CLinearOperator concept.
 *
 * Provides
 * interoperability with the Eigen API, i.e. can be used in product expression, and is usable in any
 * IterativeLinearSolver (with suitable preconditioner, i.e. the preconditioner should not be
 * constructible by analyzing matrix coefficients, since LinearOperator does not require storing any
 * matrix).
 *
 * Refer to LinearOperator.cpp for usage examples.
 *
 * @tparam ...TLinearOperators List of linear operators to be composed ordered from left to right.
 */
template <CLinearOperator... TLinearOperators>
class LinearOperator : public Eigen::EigenBase<LinearOperator<TLinearOperators...>>
{
  public:
    using SelfType = LinearOperator<TLinearOperators...>; ///< Instance type
    using BaseType = Eigen::EigenBase<SelfType>;          ///< EigenBase type

    using Scalar           = pbat::Scalar;                     ///< Eigen typedef
    using RealScalar       = pbat::Scalar;                     ///< Eigen typedef
    using StorageIndex     = typename CSCMatrix::StorageIndex; ///< Eigen typedef
    using NestedExpression = SelfType;                         ///< Eigen typedef
    enum {
        ColsAtCompileTime    = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor           = false
    };

    /**
     * @brief Construct a new Linear Operator object from a list of linear operators.
     *
     * @param inOps List of linear operators to be composed ordered from left to right.
     *
     * @note Given linear operators' lifetimes must exceed the lifetime of this LinearOperator
     * object.
     */
    LinearOperator(TLinearOperators const&... inOps);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Applies all linear operators on x, adding result to y.
     *
     * @tparam TDerivedIn Eigen matrix expression.
     * @tparam TDerivedOut Writeable Eigen matrix expression.
     * @param x Input vector or matrix.
     * @param y Output vector or matrix.
     * @pre x.rows() == InputDimensions() and y.rows() == OutputDimensions()
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Construct the matrix of all underlying matrices obtained by Lops.
     * @return CSCMatrix The compressed sparse column matrix representation of the linear operator.
     */
    CSCMatrix ToMatrix() const;

    /**
     * @brief Number of rows
     *
     * @return pbat::Index
     */
    pbat::Index OutputDimensions() const;
    /**
     * @brief Number of columns
     *
     * @return pbat::Index
     */
    pbat::Index InputDimensions() const;

    // For Eigen compatibility

    /**
     * @brief Number of rows (Eigen compatibility)
     *
     * @return BaseType::Index
     */
    BaseType::Index rows() const { return static_cast<BaseType::Index>(OutputDimensions()); }
    /**
     * @brief Number of columns (Eigen compatibility)
     *
     * @return BaseType::Index
     */
    BaseType::Index cols() const { return static_cast<BaseType::Index>(InputDimensions()); }

    /**
     * @brief Lazily left-multiply x by this linear operator.
     *
     * @tparam Rhs Right-hand side matrix or vector expression
     * @param x Right-hand side matrix or vector
     * @return Eigen::Product<SelfType, Rhs, Eigen::AliasFreeProduct> Eigen expression of the
     * product of this linear operator and x.
     */
    template <class Rhs>
    Eigen::Product<SelfType, Rhs, Eigen::AliasFreeProduct>
    operator*(Eigen::MatrixBase<Rhs> const& x) const
    {
        return Eigen::Product<SelfType, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

  private:
    std::tuple<TLinearOperators const&...> ops;
};

/**
 * @brief Compose multiple linear operators into a single linear operator.
 *
 * Refer to LinearOperator.cpp for usage examples.
 *
 * @tparam TLinearOperators Linear operator parameter types
 * @param inOps Linear operators to be composed
 * @return LinearOperator<TLinearOperators...> Composed linear operator
 */
template <CLinearOperator... TLinearOperators>
LinearOperator<TLinearOperators...> ComposeLinearOperators(TLinearOperators const&... inOps)
{
    return LinearOperator(inOps...);
}

template <CLinearOperator... TLinearOperators>
inline LinearOperator<TLinearOperators...>::LinearOperator(TLinearOperators const&... inOps)
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
inline void LinearOperator<TLinearOperators...>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.math.LinearOperator.Apply");
    std::apply([&](auto... op) { (op.Apply(x, y), ...); }, ops);
}

template <CLinearOperator... TLinearOperators>
inline CSCMatrix LinearOperator<TLinearOperators...>::ToMatrix() const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.math.LinearOperator.ToMatrix");
    CSCMatrix M(OutputDimensions(), InputDimensions());
    std::apply([&](auto... op) { ((M += op.ToMatrix()), ...); }, ops);
    return M;
}

template <CLinearOperator... TLinearOperators>
inline Index LinearOperator<TLinearOperators...>::OutputDimensions() const
{
    return std::get<0>(ops).OutputDimensions();
}

template <CLinearOperator... TLinearOperators>
inline Index LinearOperator<TLinearOperators...>::InputDimensions() const
{
    return std::get<0>(ops).InputDimensions();
}

} // namespace math
} // namespace pbat

namespace Eigen {
namespace internal {

/**
 * @brief
 *
 * See Eigen/src/Core/util/Constants.h, we want to specialize all product types.
 *
 * enum ProductImplType {
 *     DefaultProduct = 0,
 *     LazyProduct,
 *     AliasFreeProduct,
 *     CoeffBasedProductMode,
 *     LazyCoeffBasedProductMode,
 *     OuterProduct,
 *     InnerProduct,
 *     GemvProduct,
 *     GemmProduct
 * };
 *
 * This way, our matrix-free linear operators will be able to operate on all kinds of
 * matrix/vector arguments, not just dynamic sized ones, as the GemvProduct denotes.
 *
 * @tparam Rhs
 * @tparam Lhs
 */
template <pbat::math::CLinearOperator Lhs, typename Rhs, int ProductType>
struct generic_product_impl<Lhs, Rhs, SparseShape, DenseShape, ProductType>
    : generic_product_impl_base<Lhs, Rhs, generic_product_impl<Lhs, Rhs>>
{
    typedef typename Product<Lhs, Rhs>::Scalar Scalar;

    template <typename Dst>
    static void scaleAndAddTo(Dst& dst, Lhs const& lhs, Rhs const& rhs, Scalar const& alpha)
    {
        lhs.Apply(alpha * rhs, dst);
    }
};

} // namespace internal
} // namespace Eigen

#endif // PBAT_MATH_LINEAR_OPERATOR_H