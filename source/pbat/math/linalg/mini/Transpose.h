#ifndef PBAT_MATH_LINALG_MINI_TRANSPOSE_H
#define PBAT_MATH_LINALG_MINI_TRANSPOSE_H

#include "Assign.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TMatrix>
class ConstTransposeView;

template <CMatrix TMatrix>
class TransposeView;

// NOTE:
// There is a cyclic dependency between transpose and submatrix views. This is because a submatrix
// should have a transpose, but you should also be able to get a submatrix from a transposed matrix.
// To break the dependency, we create a type TransposeSubMatrix (and its const version), which has
// the exact same implementation as SubMatrix (and its const version), specifically for
// (Const)TransposeView.

template <CMatrix TMatrix, int M, int N>
class ConstTransposeSubMatrix
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = ConstTransposeSubMatrix<NestedType, M, N>;

    static auto constexpr kRows     = M;
    static auto constexpr kCols     = N;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE ConstTransposeSubMatrix(NestedType const& A, auto ib = 0, auto jb = 0)
        : A(A), ib(ib), jb(jb)
    {
        static_assert(
            NestedType::kRows >= M and NestedType::kCols >= N and M > 0 and N > 0,
            "Invalid submatrix dimensions");
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return kRows; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return kCols; }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return A(ib + i, jb + j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice(auto i, auto j) const
    {
        return ConstTransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE auto Col(auto j) const { return Slice<kRows, 1>(0, j); }
    PBAT_HOST_DEVICE auto Row(auto i) const { return Slice<1, kCols>(i, 0); }
    PBAT_HOST_DEVICE auto Transpose() const { return ConstTransposeView<SelfType>(*this); }

  private:
    NestedType const& A;
    int ib, jb;
};

template <CMatrix TMatrix, int M, int N>
class TransposeSubMatrix
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = TransposeSubMatrix<NestedType, M, N>;

    static auto constexpr kRows     = M;
    static auto constexpr kCols     = N;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE TransposeSubMatrix(NestedType& A, auto ib = 0, auto jb = 0)
        : A(A), ib(ib), jb(jb)
    {
        static_assert(
            NestedType::kRows >= M and NestedType::kCols >= N and M > 0 and N > 0,
            "Invalid submatrix dimensions");
    }

    template <class /*CMatrix*/ TOtherMatrix>
    PBAT_HOST_DEVICE SelfType& operator=(TOtherMatrix&& B)
    {
        Assign(*this, std::forward<TOtherMatrix>(B));
        return *this;
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return kRows; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return kCols; }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return A(ib + i, jb + j); }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i, auto j) { return A(ib + i, jb + j); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i) { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }
    PBAT_HOST_DEVICE ScalarType& operator[](auto i) { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice(auto i, auto j)
    {
        return TransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE auto Col(auto j) { return Slice<kRows, 1>(0, j); }
    PBAT_HOST_DEVICE auto Row(auto i) { return Slice<1, kCols>(i, 0); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice(auto i, auto j) const
    {
        return ConstTransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE auto Col(auto j) const { return Slice<kRows, 1>(0, j); }
    PBAT_HOST_DEVICE auto Row(auto i) const { return Slice<1, kCols>(i, 0); }

    PBAT_HOST_DEVICE auto Transpose() { return TransposeView<SelfType>(*this); }
    PBAT_HOST_DEVICE auto Transpose() const { return ConstTransposeView<SelfType>(*this); }

    void SetConstant(auto k) { AssignScalar(*this, k); }

  private:
    NestedType& A;
    int ib, jb;
};

template <CMatrix TMatrix>
class TransposeView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = TransposeView<NestedType>;

    static auto constexpr kRows     = NestedType::kCols;
    static auto constexpr kCols     = NestedType::kRows;
    static bool constexpr bRowMajor = not NestedType::bRowMajor;

    PBAT_HOST_DEVICE TransposeView(NestedType& A) : A(A) {}

    template <class TOtherMatrix>
    PBAT_HOST_DEVICE SelfType& operator=(TOtherMatrix&& B)
    {
        Assign(*this, std::forward<TOtherMatrix>(B));
        return *this;
    }

    PBAT_HOST_DEVICE void SetConstant(ScalarType k)
    {
        using IntegerType = std::remove_const_t<decltype(kCols)>;
        auto fRows =
            [&]<IntegerType... I>(IntegerType j, std::integer_sequence<IntegerType, I...>) {
                (((*this)(I, j) = k), ...);
            };
        auto fCols = [&]<IntegerType... J>(std::integer_sequence<IntegerType, J...>) {
            (fRows(J, std::make_integer_sequence<IntegerType, kRows>()), ...);
        };
        fCols(std::make_integer_sequence<IntegerType, kCols>());
    }

    PBAT_HOST_DEVICE constexpr auto Rows() const { return A.Cols(); }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return A.Rows(); }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return A(j, i); }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i, auto j) { return A(j, i); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i) { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }
    PBAT_HOST_DEVICE ScalarType& operator[](auto i) { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice(auto i, auto j)
    {
        return TransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE auto Col(auto j) { return Slice<kRows, 1>(0, j); }
    PBAT_HOST_DEVICE auto Row(auto i) { return Slice<1, kCols>(i, 0); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice(auto i, auto j) const
    {
        return ConstTransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE auto Col(auto j) const { return Slice<kRows, 1>(0, j); }
    PBAT_HOST_DEVICE auto Row(auto i) const { return Slice<1, kCols>(i, 0); }

    PBAT_HOST_DEVICE NestedType const& Transpose() const { return A; }
    PBAT_HOST_DEVICE NestedType& Transpose() { return A; }

    void SetConstant(auto k) { AssignScalar(*this, k); }

  private:
    NestedType& A;
};

template <CMatrix TMatrix>
class ConstTransposeView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = ConstTransposeView<NestedType>;

    static auto constexpr kRows     = NestedType::kCols;
    static auto constexpr kCols     = NestedType::kRows;
    static bool constexpr bRowMajor = not NestedType::bRowMajor;

    PBAT_HOST_DEVICE ConstTransposeView(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return A.Cols(); }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return A.Rows(); }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return A(j, i); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice(auto i, auto j) const
    {
        return ConstTransposeSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE auto Col(auto j) const { return Slice<kRows, 1>(0, j); }
    PBAT_HOST_DEVICE auto Row(auto i) const { return Slice<1, kCols>(i, 0); }

    PBAT_HOST_DEVICE NestedType const& Transpose() const { return A; }

  private:
    NestedType const& A;
};

#define PBAT_MINI_TRANSPOSE_API(SelfType)      \
    PBAT_HOST_DEVICE auto Transpose()          \
    {                                          \
        return TransposeView<SelfType>(*this); \
    }

#define PBAT_MINI_CONST_TRANSPOSE_API(SelfType)     \
    PBAT_HOST_DEVICE auto Transpose() const         \
    {                                               \
        return ConstTransposeView<SelfType>(*this); \
    }

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_TRANSPOSE_H