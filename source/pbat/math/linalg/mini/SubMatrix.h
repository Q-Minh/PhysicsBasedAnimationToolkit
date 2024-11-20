#ifndef PBAT_MATH_LINALG_MINI_SUBMATRIX_H
#define PBAT_MATH_LINALG_MINI_SUBMATRIX_H

#include "Assign.h"
#include "Concepts.h"
#include "Transpose.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix, int M, int N>
class ConstSubMatrix
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = ConstSubMatrix<NestedType, M, N>;

    static auto constexpr kRows     = M;
    static auto constexpr kCols     = N;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE ConstSubMatrix(NestedType const& A, int ib, int jb) : A(A), ib(ib), jb(jb)
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
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, kRows, 1> Col(auto j) const
    {
        return Slice<kRows, 1>(0, j);
    }
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, 1, kCols> Row(auto i) const
    {
        return Slice<1, kCols>(i, 0);
    }
    PBAT_HOST_DEVICE ConstTransposeView<SelfType const> Transpose() const
    {
        return ConstTransposeView<SelfType const>(*this);
    }

  private:
    NestedType const& A;
    int ib, jb;
};

template <class /*CMatrix*/ TMatrix, int M, int N>
class SubMatrix
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = SubMatrix<NestedType, M, N>;

    static auto constexpr kRows     = M;
    static auto constexpr kCols     = N;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE SubMatrix(NestedType& A, int ib, int jb) : A(A), ib(ib), jb(jb)
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

    PBAT_HOST_DEVICE auto Rows() const { return kRows; }
    PBAT_HOST_DEVICE auto Cols() const { return kCols; }

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
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE auto Col(auto j) { return Slice<kRows, 1>(0, j); }
    PBAT_HOST_DEVICE auto Row(auto i) { return Slice<1, kCols>(i, 0); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice(auto i, auto j) const
    {
        return ConstSubMatrix<SelfType, S, T>(*this, i, j);
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

#define PBAT_MINI_SUBMATRIX_API(SelfType)                      \
    template <auto S, auto T>                                  \
    PBAT_HOST_DEVICE [[maybe_unused]] auto Slice(int i, int j) \
    {                                                          \
        return SubMatrix<SelfType, S, T>(*this, i, j);         \
    }                                                          \
    PBAT_HOST_DEVICE [[maybe_unused]] auto Col(int j)          \
    {                                                          \
        return Slice<kRows, 1>(0, j);                          \
    }                                                          \
    PBAT_HOST_DEVICE [[maybe_unused]] auto Row(int i)          \
    {                                                          \
        return Slice<1, kCols>(i, 0);                          \
    }

#define PBAT_MINI_CONST_SUBMATRIX_API(SelfType)                      \
    template <auto S, auto T>                                        \
    PBAT_HOST_DEVICE [[maybe_unused]] auto Slice(int i, int j) const \
    {                                                                \
        return ConstSubMatrix<SelfType, S, T>(*this, i, j);          \
    }                                                                \
    PBAT_HOST_DEVICE [[maybe_unused]] auto Col(int j) const          \
    {                                                                \
        return Slice<kRows, 1>(0, j);                                \
    }                                                                \
    PBAT_HOST_DEVICE [[maybe_unused]] auto Row(int i) const          \
    {                                                                \
        return Slice<1, kCols>(i, 0);                                \
    }

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_SUBMATRIX_H