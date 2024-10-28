#ifndef PBAT_MATH_LINALG_MINI_EIGEN_H
#define PBAT_MATH_LINALG_MINI_EIGEN_H

#include "Api.h"
#include "Concepts.h"

#include <Eigen/Core>
#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class TMatrix>
concept CEigenConvertible = CMatrix<TMatrix> && requires(TMatrix M)
{
    {
        M.Data()
    } -> std::convertible_to<typename TMatrix::ScalarType*>;
};

template <class TMatrix>
Eigen::Map<Eigen::Matrix<
    typename std::remove_cvref_t<TMatrix>::ScalarType,
    std::remove_cvref_t<TMatrix>::kRows,
    std::remove_cvref_t<TMatrix>::kCols,
    (std::remove_cvref_t<TMatrix>::bRowMajor) ? Eigen::RowMajor : Eigen::ColMajor>>
ToEigen(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CEigenConvertible<MatrixType>, "A must satisfy CEigenConvertible");
    using EigenMatrixType = Eigen::Matrix<
        typename MatrixType::ScalarType,
        MatrixType::kRows,
        MatrixType::kCols,
        (MatrixType::bRowMajor) ? Eigen::RowMajor : Eigen::ColMajor>;
    return Eigen::Map<EigenMatrixType>(A.Data());
}

template <class TDerived>
class ConstEigenMatrixWrapper
{
  public:
    using ScalarType = typename TDerived::Scalar;
    using SelfType   = ConstEigenMatrixWrapper<TDerived>;

    static int constexpr kRows      = TDerived::RowsAtCompileTime;
    static int constexpr kCols      = TDerived::ColsAtCompileTime;
    static bool constexpr bRowMajor = TDerived::IsRowMajor;

    PBAT_HOST_DEVICE ConstEigenMatrixWrapper(TDerived const& A) : mA(A)
    {
        static_assert(TDerived::RowsAtCompileTime > 0, "A must have compile-time row dimensions");
        static_assert(
            TDerived::ColsAtCompileTime > 0,
            "A must have compile-time column dimensions");
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return mA(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return mA.reshaped()(i); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    TDerived const& mA;
};

template <class TDerived>
class EigenMatrixWrapper
{
  public:
    using ScalarType = typename TDerived::Scalar;
    using SelfType   = EigenMatrixWrapper<TDerived>;

    static int constexpr kRows      = TDerived::RowsAtCompileTime;
    static int constexpr kCols      = TDerived::ColsAtCompileTime;
    static bool constexpr bRowMajor = TDerived::IsRowMajor;

    PBAT_HOST_DEVICE EigenMatrixWrapper(TDerived& A) : mA(A)
    {
        static_assert(TDerived::RowsAtCompileTime > 0, "A must have compile-time row dimensions");
        static_assert(
            TDerived::ColsAtCompileTime > 0,
            "A must have compile-time column dimensions");
    }

    PBAT_HOST_DEVICE void SetConstant(ScalarType k) { Assign(*this, k); }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return mA(i, j); }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i, auto j) { return mA(i, j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return mA.reshaped()(i); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i) { return mA.reshaped()(i); }
    PBAT_HOST_DEVICE ScalarType& operator[](auto i) { return (*this)(i); }

    PBAT_MINI_READ_WRITE_API(SelfType)

  private:
    TDerived& mA;
};

template <class TDerived>
auto FromEigen(TDerived& A)
{
    return EigenMatrixWrapper<TDerived>(A);
}

template <class TDerived>
auto FromEigen(TDerived const& A)
{
    return ConstEigenMatrixWrapper<TDerived>(A);
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_EIGEN_H