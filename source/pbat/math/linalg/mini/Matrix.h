#ifndef PBAT_MATH_LINALG_MINI_MATRIX_CUH
#define PBAT_MATH_LINALG_MINI_MATRIX_CUH

#include "Api.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"
#include "pbat/common/ConstexprFor.h"

#include <array>
#include <cstdint>
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

template <class TScalar, int M>
PBAT_HOST_DEVICE auto Unit(auto i)
{
    return Identity<TScalar, M, M>().Col(i);
}

template <int M, int N, class TScalar>
PBAT_HOST_DEVICE auto FromFlatBuffer(TScalar* buf, std::int64_t bi)
{
    return SMatrixView<TScalar, M, N>(buf + M * N * bi);
}

template <class TScalar, CMatrix TIndexMatrix>
PBAT_HOST_DEVICE auto FromFlatBuffer(TScalar* buf, TIndexMatrix const& inds)
{
    using IntegerType = typename TIndexMatrix::ScalarType;
    static_assert(std::is_integral_v<IntegerType>, "inds must be matrix of indices");
    auto constexpr M = TIndexMatrix::kRows;
    auto constexpr N = TIndexMatrix::kCols;
    using ScalarType = std::remove_cvref_t<TScalar>;
    SMatrix<ScalarType, M, N> A{};
    using pbat::common::ForRange;
    ForRange<0, N>([&]<auto j>() { ForRange<0, M>([&]<auto i>() { A(i, j) = buf[inds(i, j)]; }); });
    return A;
}

template <CMatrix TMatrix>
PBAT_HOST_DEVICE void
ToFlatBuffer(TMatrix const& A, typename TMatrix::ScalarType* buf, std::int64_t bi)
{
    auto constexpr M              = TMatrix::kRows;
    auto constexpr N              = TMatrix::kCols;
    FromFlatBuffer<M, N>(buf, bi) = A;
}

template <CMatrix TMatrix, CMatrix TIndexMatrix>
PBAT_HOST_DEVICE void
ToFlatBuffer(TMatrix const& A, TIndexMatrix const& inds, typename TMatrix::ScalarType* buf)
{
    auto constexpr MA = TMatrix::kRows;
    auto constexpr NA = TMatrix::kCols;
    auto constexpr MI = TIndexMatrix::kRows;
    auto constexpr NI = TIndexMatrix::kCols;
    static_assert(MA == MI or MI == 1, "A must have same rows as inds or inds is a row vector");
    static_assert(NA == NI, "A must have same cols as inds");
    using pbat::common::ForRange;
    if constexpr (MA > 1 and MI == 1)
    {
        // In this case, I will assume that the user wishes to put each column of A in the
        // corresponding "column" in the flat buffer buf, as if column major, according to inds.
        ForRange<0, NA>([&]<auto j>() {
            ForRange<0, MA>([&]<auto i>() { buf[MA * inds(0, j) + i] = A(i, j); });
        });
    }
    else
    {
        ForRange<0, NA>(
            [&]<auto j>() { ForRange<0, MA>([&]<auto i>() { buf[inds(i, j)] = A(i, j); }); });
    }
}

template <int M, int N, class TScalar>
PBAT_HOST_DEVICE auto FromBuffers(std::array<TScalar*, M> buf, std::int64_t bi)
{
    using ScalarType = std::remove_const_t<TScalar>;
    SMatrix<ScalarType, M, N> A{};
    using pbat::common::ForRange;
    ForRange<0, M>([&]<auto i>() { A.Row(i) = FromFlatBuffer<1, N>(buf[i], bi); });
    return A;
}

template <int K, class TScalar, CMatrix TIndexMatrix>
PBAT_HOST_DEVICE auto FromBuffers(std::array<TScalar*, K> buf, TIndexMatrix const& inds)
{
    using IntegerType = typename TIndexMatrix::ScalarType;
    static_assert(std::is_integral_v<IntegerType>, "inds must be matrix of indices");
    auto constexpr M = TIndexMatrix::kRows;
    auto constexpr N = TIndexMatrix::kCols;
    using ScalarType = std::remove_cvref_t<TScalar>;
    SMatrix<ScalarType, K * M, N> A{};
    using pbat::common::ForRange;
    ForRange<0, K>(
        [&]<auto k>() { A.template Slice<M, N>(k * M, 0) = FromFlatBuffer(buf[k], inds); });
    return A;
}

template <CMatrix TMatrix, int M>
PBAT_HOST_DEVICE void
ToBuffers(TMatrix const& A, std::array<typename TMatrix::ScalarType*, M> buf, std::int64_t bi)
{
    static_assert(M == TMatrix::kRows, "A must have same rows as number of buffers");
    auto constexpr N = TMatrix::kCols;
    using pbat::common::ForRange;
    ForRange<0, M>([&]<auto i>() { FromFlatBuffer<1, N>(buf[i], bi) = A.Row(i); });
}

template <CMatrix TMatrix, CMatrix TIndexMatrix, int K>
PBAT_HOST_DEVICE void ToBuffers(
    TMatrix const& A,
    TIndexMatrix const& inds,
    std::array<typename TMatrix::ScalarType*, K> buf)
{
    auto constexpr MA = TMatrix::kRows;
    auto constexpr NA = TMatrix::kCols;
    auto constexpr MI = TIndexMatrix::kRows;
    auto constexpr NI = TIndexMatrix::kCols;
    static_assert(MA % MI == 0, "Rows of A must be multiple of rows of inds");
    static_assert(NA == NI, "A and inds must have same number of columns");
    static_assert(MA / MI == K, "A must have number of rows == #buffers*#rows of inds");
    using ScalarType = typename TMatrix::ScalarType;
    using pbat::common::ForRange;
    ForRange<0, K>(
        [&]<auto k>() { ToFlatBuffer(A.template Slice<MI, NI>(k * MI, 0), inds, buf[k]); });
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_MATRIX_CUH