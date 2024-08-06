#ifndef PBAT_GPU_MATH_LINALG_MATRIX_CUH
#define PBAT_GPU_MATH_LINALG_MATRIX_CUH

#include <array>
#include <concepts>

namespace pbat {
namespace gpu {
namespace math {
namespace linalg {

template <class TMatrix>
concept CMatrix = requires(TMatrix M)
{
    requires std::is_arithmetic_v<typename TMatrix::ScalarType>;
    {
        M.Rows()
    } -> std::convertible_to<int>;
    {
        M.Cols()
    } -> std::convertible_to<int>;
    {
        TMatrix::kRows
    } -> std::convertible_to<int>;
    {
        TMatrix::kCols
    } -> std::convertible_to<int>;
    {
        M(0, 0)
    } -> std::convertible_to<typename TMatrix::ScalarType>;
};

template <CMatrix TMatrix>
class TransposeView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;

    static auto constexpr kRows = TMatrix::kCols;
    static auto constexpr kCols = TMatrix::kRows;

    TransposeView(NestedType const& A) : A(A) {}
    constexpr auto Rows() const { return A.Cols(); }
    constexpr auto Cols() const { return A.Rows(); }

    auto operator()(auto i, auto j) const { return A(j, i); }
    auto& operator()(auto i, auto j) { return A(j, i); }

    NestedType const& Transpose() const { return A; }
    NestedType& Transpose() { return A; }

  private:
    NestedType const& A;
};

template <CMatrix TMatrix, int M, int N>
class SubMatrix
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = SubMatrix<NestedType, M, N>;

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    SubMatrix(NestedType const& A, auto ib = 0, auto jb = 0) : A(A), ib(ib), jb(jb)
    {
        static_assert(
            A.Rows() >= M and A.Cols() >= N and M > 0 and N > 0,
            "Invalid submatrix dimensions");
    }

    constexpr auto Rows() const { return kRows; }
    constexpr auto Cols() const { return kCols; }

    auto operator()(auto i, auto j) const { return A(ib + i, jb + j); }
    auto& operator()(auto i, auto j) { return A(ib + i, jb + j); }

    template <auto S, auto T>
    SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    TransposeView<SelfType> Transpose() const { return TransposeView<SelfType>(*this); }

  private:
    NestedType const& A;
    int ib, jb;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Sum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Sum<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows = LhsNestedType::kRows;
    static auto constexpr kCols = RhsNestedType::kCols;

    Sum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "Invalid matrix sum dimensions");
    }

    constexpr auto Rows() const { return kRows; }
    constexpr auto Cols() const { return kCols; }

    auto operator()(auto i, auto j) const { return A(i, j) + B(i, j); }

    template <auto S, auto T>
    SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    TransposeView<SelfType> Transpose() const { return TransposeView<SelfType>(*this); }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TMatrix>
class Scale
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = Scale<NestedType>;

    static auto constexpr kRows = NestedType::kRows;
    static auto constexpr kCols = NestedType::kCols;

    Scale(ScalarType k, NestedType const& A) : k(k), A(A) {}

    constexpr auto Rows() const { return kRows; }
    constexpr auto Cols() const { return kCols; }

    auto operator()(auto i, auto j) const { return k * A(i, j); }

    template <auto S, auto T>
    SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    TransposeView<SelfType> Transpose() const { return TransposeView<SelfType>(*this); }

  private:
    ScalarType k;
    NestedType const& A;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Product
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = LhsNestedType::ScalarType;
    using SelfType   = Product<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows = LhsNestedType::kRows;
    static auto constexpr kCols = RhsNestedType::kCols;

    Product(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B) {}

    constexpr auto Rows() const { return kRows; }
    constexpr auto Cols() const { return kCols; }

    auto operator()(auto i, auto j) const
    {
        ScalarType aij{0.};
        for (auto k = 0; k < A.Cols(); ++k)
            aij += A(i, k) * B(k, j);
        return aij;
    }

    template <auto S, auto T>
    SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    TransposeView<SelfType> Transpose() const { return TransposeView<SelfType>(*this); }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class CrossProduct
{
  public:
    static_assert(
        ((TLhsMatrix::kRows == 3 and TLhsMatrix::kCols == 1) or
         (TLhsMatrix::kRows == 1 and TLhsMatrix::kCols == 3)) and
            ((TRhsMatrix::kRows == 3 and TRhsMatrix::kCols == 1) or
             (TRhsMatrix::kRows == 1 and TRhsMatrix::kCols == 3)),
        "Cross product only valid for 3x1 or 1x3 matrices");

    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = LhsNestedType::ScalarType;
    using SelfType   = CrossProduct<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows = 3;
    static auto constexpr kCols = 1;

    CrossProduct(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B) {}

    constexpr auto Rows() const { return kRows; }
    constexpr auto Cols() const { return kCols; }

    auto operator()(auto i, auto j) const
    {
        j      = (i + 1) % 3;
        auto k = (i + 2) % 3;
        return A(j, 0) * B(k, 0) - A(k, 0) * B(j, 0);
    }

    template <auto S, auto T>
    SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    TransposeView<SelfType> Transpose() const { return TransposeView<SelfType>(*this); }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <class TScalar, int M, int N>
class Matrix
{
  public:
    using ScalarType = TScalar;
    using SelfType   = Matrix<ScalarType, M, N>;

    Matrix() : a() {}

    template <CMatrix TMatrix>
    Matrix(TMatrix const& B) : a()
    {
        static_assert(
            TMatrix::kRows == kRows and TMatrix::kCols == kCols,
            "Invalid matrix assignment dimensions");
        for (auto j = 0; j < kCols; ++j)
            for (auto i = 0; i < kRows; ++i)
                (*this)(i, j) = B(i, j);
    }

    template <CMatrix TMatrix>
    SelfType& operator=(TMatrix const& B)
    {
        for (auto j = 0; j < kCols; ++j)
            for (auto i = 0; i < kRows; ++i)
                (*this)(i, j) = B(i, j);
    }

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    constexpr auto Rows() const { return kRows; }
    constexpr auto Cols() const { return kCols; }

    auto operator()(auto i, auto j) const { return a[j * M + i]; }
    auto& operator()(auto i, auto j) { return a[j * M + i]; }

    // Vector(ized) access
    auto operator()(auto i) const { return a[i]; }
    auto& operator()(auto i) { return a[i]; }

    // Smart accessors
    template <auto S, auto T>
    SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    TransposeView<SelfType> Transpose() const { return TransposeView<SelfType>(*this); }

    void SetZero() { memset(a.data(), 0, kRows * kCols * sizeof(ScalarType)); }

  private:
    std::array<ScalarType, M * N> a;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
auto operator+(TLhsMatrix const& A, TRhsMatrix const& B)
{
    return Sum<TLhsMatrix, TRhsMatrix>(A, B);
}

template <CMatrix TMatrix>
auto operator-(TMatrix const& A)
{
    using ScalarType = typename TMatrix::ScalarType;
    return Scale<TMatrix>(ScalarType(-1.), A);
}

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
auto operator-(TLhsMatrix const& A, TRhsMatrix const& B)
{
    using NegatedMatrixType = Scale<TRhsMatrix>;
    NegatedMatrixType negB  = -B;
    return Sum<TLhsMatrix, NegatedMatrixType>(A, negB);
}

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
auto operator*(TLhsMatrix const& A, TRhsMatrix const& B)
{
    return Product<TLhsMatrix, TRhsMatrix>(A, B);
}

template <CMatrix TMatrix>
auto operator*(typename TMatrix::ScalarType k, TMatrix const& A)
{
    return Scale<TMatrix>(k, A);
}

template <CMatrix TMatrix>
auto operator/(TMatrix const& A, typename TMatrix::ScalarType k)
{
    using ScalarType = typename TMatrix::ScalarType;
    return Scale<TMatrix>(ScalarType(1. / k), A);
}

template <CMatrix TMatrix>
auto Trace(TMatrix const& A)
{
    static_assert(A.Rows() == A.Cols(), "Cannot compute trace of non-square matrix");
    typename TMatrix::ScalarType tr{0.};
    for (auto i = 0; i < A.Cols(); ++i)
        tr += A(i, i);
    return tr;
}

template <CMatrix TMatrix>
auto SquaredNorm(TMatrix const& A)
{
    return Trace(A.Transpose() * A());
}

template <CMatrix TMatrix>
auto Norm(TMatrix const& A)
{
    using ScalarType = typename TMatrix::ScalarType;
    if constexpr (std::is_same_v<ScalarType, float>)
    {
        return sqrtf(Trace(A));
    }
    else
    {
        return sqrt(Trace(A));
    }
}

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
auto Cross(TLhsMatrix const& A, TRhsMatrix const& B)
{
    return CrossProduct<TLhsMatrix, TRhsMatrix>(A, B);
}

template <CMatrix TMatrix>
auto Determinant(TMatrix const& A)
{
    static_assert(A.Rows() == A.Cols(), "Cannot compute determinant of non-square matrix");
    static_assert(A.Rows() < 4, "Determinant of matrix of dimensions >= 4 too costly");
    if constexpr (A.Rows() == 1)
    {
        return A(0, 0);
    }
    if constexpr (A.Rows() == 2)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }
    if constexpr (A.Rows() == 3)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
               A(0, 1) * (A(1, 0) * A(2, 2) - A(2, 0) * A(1, 2)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1));
    }
}

} // namespace linalg
} // namespace math
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_MATH_LINALG_MATRIX_CUH