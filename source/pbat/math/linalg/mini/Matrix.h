#ifndef PBAT_MATH_LINALG_MINI_MATRIX_CUH
#define PBAT_MATH_LINALG_MINI_MATRIX_CUH

#include "Api.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <array>
#include <initializer_list>
#include <string.h>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class TScalar, int M, int N>
class Ones
{
  public:
    using ScalarType = TScalar;
    using SelfType   = Ones<ScalarType, M, N>;

    static auto constexpr kRows     = M;
    static auto constexpr kCols     = N;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE ScalarType operator()([[maybe_unused]] auto i, [[maybe_unused]] auto j) const
    {
        return ScalarType{1};
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()([[maybe_unused]] auto i) const { return ScalarType{1}; }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice([[maybe_unused]] auto i, [[maybe_unused]] auto j) const
    {
        return Ones<ScalarType, S, T>();
    }
    PBAT_HOST_DEVICE auto Col([[maybe_unused]] auto j) const
    {
        return Ones<ScalarType, kRows, 1>();
    }
    PBAT_HOST_DEVICE auto Row([[maybe_unused]] auto i) const
    {
        return Ones<ScalarType, 1, kCols>();
    }

    PBAT_MINI_DIMENSIONS_API
    PBAT_MINI_CONST_TRANSPOSE_API(SelfType)
};

template <class TScalar, int M, int N>
class Zeros
{
  public:
    using ScalarType = TScalar;
    using SelfType   = Zeros<ScalarType, M, N>;

    static auto constexpr kRows     = M;
    static auto constexpr kCols     = N;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE ScalarType operator()([[maybe_unused]] auto i, [[maybe_unused]] auto j) const
    {
        return ScalarType{0};
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()([[maybe_unused]] auto i) const { return ScalarType{0}; }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE auto Slice([[maybe_unused]] auto i, [[maybe_unused]] auto j) const
    {
        return Zeros<ScalarType, S, T>();
    }
    PBAT_HOST_DEVICE auto Col([[maybe_unused]] auto j) const
    {
        return Zeros<ScalarType, kRows, 1>();
    }
    PBAT_HOST_DEVICE auto Row([[maybe_unused]] auto i) const
    {
        return Zeros<ScalarType, 1, kCols>();
    }

    PBAT_MINI_DIMENSIONS_API
    PBAT_MINI_CONST_TRANSPOSE_API(SelfType)
};

template <class TScalar, int M, int N>
class Identity
{
  public:
    using ScalarType = TScalar;
    using SelfType   = Identity<ScalarType, M, N>;

    static auto constexpr kRows     = M;
    static auto constexpr kCols     = N;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        return static_cast<ScalarType>(i == j);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)
};

template <class TScalar, int M, int N = 1>
class SMatrix
{
  public:
    using ScalarType  = TScalar;
    using SelfType    = SMatrix<ScalarType, M, N>;
    using StorageType = std::array<ScalarType, M * N>;
    using IndexType   = typename StorageType::size_type;

    PBAT_HOST_DEVICE SMatrix() : a() {}

    template <class... T>
    PBAT_HOST_DEVICE SMatrix(T... values) : a{values...}
    {
    }

    static int constexpr kRows      = M;
    static int constexpr kCols      = N;
    static bool constexpr bRowMajor = false;

    template <class /*CMatrix*/ TMatrix>
    PBAT_HOST_DEVICE SMatrix(TMatrix&& B) : a()
    {
        Assign(*this, std::forward<TMatrix>(B));
    }

    PBAT_HOST_DEVICE void SetConstant(ScalarType k) { AssignScalar(*this, k); }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        auto k = static_cast<IndexType>(j * M + i);
        return a[k];
    }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i, auto j)
    {
        auto k = static_cast<IndexType>(j * M + i);
        return a[k];
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const
    {
        auto k = static_cast<IndexType>(i);
        return a[k];
    }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i)
    {
        auto k = static_cast<IndexType>(i);
        return a[k];
    }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const
    {
        auto k = static_cast<IndexType>(i);
        return a[k];
    }
    PBAT_HOST_DEVICE ScalarType& operator[](auto i)
    {
        auto k = static_cast<IndexType>(i);
        return a[k];
    }

    PBAT_HOST_DEVICE void SetZero() { memset(a.data(), 0, kRows * kCols * sizeof(ScalarType)); }

    ScalarType* Data() { return a.data(); }
    ScalarType const* Data() const { return a.data(); }

    PBAT_MINI_READ_WRITE_API(SelfType)

  private:
    StorageType a;
};

template <class TScalar, int M>
using SVector = SMatrix<TScalar, M, 1>;

template <class TScalar, int M, int N>
class SMatrixView
{
  public:
    using ScalarType = TScalar;
    using SelfType   = SMatrixView<ScalarType, M, N>;

    static auto constexpr kRows     = M;
    static auto constexpr kCols     = N;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE SMatrixView(ScalarType* a) : mA(a) {}

    PBAT_HOST_DEVICE void SetConstant(ScalarType k) { AssignScalar(*this, k); }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const { return mA[j * M + i]; }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i, auto j) { return mA[j * M + i]; }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return mA[i]; }
    PBAT_HOST_DEVICE ScalarType& operator()(auto i) { return mA[i]; }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return mA[i]; }
    PBAT_HOST_DEVICE ScalarType& operator[](auto i) { return mA[i]; }

    PBAT_HOST_DEVICE void SetZero() { memset(mA, 0, kRows * kCols * sizeof(ScalarType)); }

    ScalarType* Data() { return mA; }
    ScalarType const* Data() const { return mA; }

    PBAT_MINI_READ_WRITE_API(SelfType)

  private:
    ScalarType* mA;
};

template <class TScalar, int M>
using SVectorView = SMatrixView<TScalar, M, 1>;

template <CMatrix TMatrix, int RepeatRows, int RepeatCols>
class TiledView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = TiledView<NestedType, RepeatRows, RepeatCols>;

    static auto constexpr kRows     = RepeatRows * NestedType::kRows;
    static auto constexpr kCols     = RepeatCols * NestedType::kCols;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE TiledView(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        return A(i % NestedType::kRows, j % NestedType::kCols);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    NestedType const& A;
};

template <auto RepeatRows, auto RepeatCols, class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Repeat(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    return TiledView<MatrixType, RepeatRows, RepeatCols>(std::forward<TMatrix>(A));
}

template <class TScalar, int M>
PBAT_HOST_DEVICE auto Unit(auto i)
{
    return Identity<TScalar, M, M>().Col(i);
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_MATRIX_CUH