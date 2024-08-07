#ifndef PBAT_GPU_MATH_LINALG_MATRIX_CUH
#define PBAT_GPU_MATH_LINALG_MATRIX_CUH

#include <array>
#include <concepts>
#include <utility>

namespace pbat {
namespace gpu {
namespace math {
namespace linalg {

template <class TMatrix>
concept CMatrix = requires(TMatrix M)
{
    typename TMatrix::ScalarType;
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

template <class /*CMatrix*/ TMatrix>
class TransposeView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = TransposeView<NestedType>;

    static auto constexpr kRows = NestedType::kCols;
    static auto constexpr kCols = NestedType::kRows;

    __host__ __device__ TransposeView(NestedType& A) : A(A) {}

    template <class /*CMatrix*/ TMatrix>
    __host__ __device__ SelfType& operator=(TMatrix&& B)
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
        return *this;
    }

    __host__ __device__ constexpr auto Rows() const { return A.Cols(); }
    __host__ __device__ constexpr auto Cols() const { return A.Rows(); }

    __host__ __device__ auto operator()(auto i, auto j) const { return A(j, i); }
    __host__ __device__ auto& operator()(auto i, auto j) { return A(j, i); }

    __host__ __device__ NestedType const& Transpose() const { return A; }
    __host__ __device__ NestedType& Transpose() { return A; }

  private:
    NestedType& A;
};

template <class /*CMatrix*/ TMatrix, int M, int N>
class SubMatrix
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = SubMatrix<NestedType, M, N>;

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    __host__ __device__ SubMatrix(NestedType& A, auto ib = 0, auto jb = 0) : A(A), ib(ib), jb(jb)
    {
        static_assert(
            NestedType::kRows >= M and NestedType::kCols >= N and M > 0 and N > 0,
            "Invalid submatrix dimensions");
    }

    template <class /*CMatrix*/ TMatrix>
    __host__ __device__ SelfType& operator=(TMatrix&& B)
    {
        static_assert(
            TMatrix::kRows == kRows and TMatrix::kCols == kCols,
            "Invalid submatrix dimensions");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
        return *this;
    }

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return A(ib + i, jb + j); }
    __host__ __device__ auto& operator()(auto i, auto j) { return A(ib + i, jb + j); }

    template <auto S, auto T>
    __host__ __device__ SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    __host__ __device__ SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    __host__ __device__ TransposeView<SelfType> Transpose()
    {
        return TransposeView<SelfType>(*this);
    }
    __host__ __device__ TransposeView<SelfType const> Transpose() const
    {
        return TransposeView<SelfType const>(*this);
    }

  private:
    NestedType& A;
    int ib, jb;
};

template <class /*CMatrix*/ TMatrix, int M, int N>
class ConstSubMatrix
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = ConstSubMatrix<NestedType, M, N>;

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    __host__ __device__ ConstSubMatrix(NestedType const& A, auto ib = 0, auto jb = 0)
        : A(A), ib(ib), jb(jb)
    {
        static_assert(
            NestedType::kRows >= M and NestedType::kCols >= N and M > 0 and N > 0,
            "Invalid submatrix dimensions");
    }

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return A(ib + i, jb + j); }

    template <auto S, auto T>
    __host__ __device__ ConstSubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return ConstSubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ ConstSubMatrix<SelfType, kRows, 1> Col(auto j)
    {
        return Slice<kRows, 1>(0, j);
    }
    __host__ __device__ ConstSubMatrix<SelfType, 1, kCols> Row(auto i)
    {
        return Slice<1, kCols>(i, 0);
    }
    __host__ __device__ TransposeView<SelfType> Transpose()
    {
        return TransposeView<SelfType>(*this);
    }
    __host__ __device__ TransposeView<SelfType const> Transpose() const
    {
        return TransposeView<SelfType const>(*this);
    }

  private:
    NestedType const& A;
    int ib, jb;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
class Sum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Sum<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows = LhsNestedType::kRows;
    static auto constexpr kCols = RhsNestedType::kCols;

    __host__ __device__ Sum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "Invalid matrix sum dimensions");
    }

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return A(i, j) + B(i, j); }

    template <auto S, auto T>
    __host__ __device__ SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    __host__ __device__ SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    __host__ __device__ TransposeView<SelfType> Transpose() const
    {
        return TransposeView<SelfType>(*this);
    }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <class /*CMatrix*/ TMatrix>
class Scale
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = Scale<NestedType>;

    static auto constexpr kRows = NestedType::kRows;
    static auto constexpr kCols = NestedType::kCols;

    __host__ __device__ Scale(ScalarType k, NestedType const& A) : k(k), A(A) {}

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return k * A(i, j); }

    template <auto S, auto T>
    __host__ __device__ SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    __host__ __device__ SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    __host__ __device__ TransposeView<SelfType> Transpose() const
    {
        return TransposeView<SelfType>(*this);
    }

  private:
    ScalarType k;
    NestedType const& A;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
class Product
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = LhsNestedType::ScalarType;
    using SelfType   = Product<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows = LhsNestedType::kRows;
    static auto constexpr kCols = RhsNestedType::kCols;

    __host__ __device__ Product(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B) {}

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const
    {
        auto contract = [this, i, j]<auto... K>(std::index_sequence<K...>) {
            return ((A(i, K) * B(K, j)) + ...);
        };
        return contract(std::make_index_sequence<LhsNestedType::kCols>());
    }

    template <auto S, auto T>
    __host__ __device__ SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    __host__ __device__ SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    __host__ __device__ TransposeView<SelfType> Transpose() const
    {
        return TransposeView<SelfType>(*this);
    }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
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

    __host__ __device__ CrossProduct(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B) {}

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const
    {
        j      = (i + 1) % 3;
        auto k = (i + 2) % 3;
        return A(j, 0) * B(k, 0) - A(k, 0) * B(j, 0);
    }

    template <auto S, auto T>
    __host__ __device__ SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    __host__ __device__ SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    __host__ __device__ TransposeView<SelfType> Transpose() const
    {
        return TransposeView<SelfType>(*this);
    }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <class TScalar, int M, int N = 1>
class Matrix
{
  public:
    using ScalarType = TScalar;
    using SelfType   = Matrix<ScalarType, M, N>;

    __host__ __device__ Matrix() : a() {}

    template <class /*CMatrix*/ TMatrix>
    __host__ __device__ Matrix(TMatrix&& B) : a()
    {
        static_assert(
            TMatrix::kRows == kRows and TMatrix::kCols == kCols,
            "Invalid matrix assignment dimensions");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
    }

    template <class /*CMatrix*/ TMatrix>
    __host__ __device__ SelfType& operator=(TMatrix&& B)
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            ((*this)(I, j) = std::forward<TMatrix>(B)(I, j), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
        return *this;
    }

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return this->a[j * M + i]; }
    __host__ __device__ auto& operator()(auto i, auto j) { return this->a[j * M + i]; }

    // Vector(ized) access
    __host__ __device__ auto operator()(auto i) const { return a[i]; }
    __host__ __device__ auto& operator()(auto i) { return a[i]; }

    // Smart accessors
    template <auto S, auto T>
    __host__ __device__ SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    template <auto S, auto T>
    __host__ __device__ ConstSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstSubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    __host__ __device__ SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    __host__ __device__ ConstSubMatrix<SelfType, kRows, 1> Col(auto j) const
    {
        return Slice<kRows, 1>(0, j);
    }
    __host__ __device__ ConstSubMatrix<SelfType, 1, kCols> Row(auto i) const
    {
        return Slice<1, kCols>(i, 0);
    }
    __host__ __device__ TransposeView<SelfType> Transpose()
    {
        return TransposeView<SelfType>(*this);
    }
    __host__ __device__ TransposeView<SelfType const> Transpose() const
    {
        return TransposeView<SelfType const>(*this);
    }

    void SetZero() { memset(a.data(), 0, kRows * kCols * sizeof(ScalarType)); }

  private:
    std::array<ScalarType, M * N> a;
};

template <class TScalar, int M, int N>
class MatrixView
{
  public:
    using ScalarType = TScalar;
    using SelfType   = MatrixView<ScalarType, M, N>;

    __host__ __device__ MatrixView(ScalarType* a) : a(a) {}

    template <class /*CMatrix*/ TMatrix>
    __host__ __device__ SelfType& operator=(TMatrix&& B)
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
        return *this;
    }

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return a[j * M + i]; }
    __host__ __device__ auto& operator()(auto i, auto j) { return a[j * M + i]; }

    // Vector(ized) access
    __host__ __device__ auto operator()(auto i) const { return a[i]; }
    __host__ __device__ auto& operator()(auto i) { return a[i]; }

    // Smart accessors
    template <auto S, auto T>
    __host__ __device__ SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    __host__ __device__ SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }

    template <auto S, auto T>
    __host__ __device__ ConstSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstSubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ ConstSubMatrix<SelfType, kRows, 1> Col(auto j) const
    {
        return Slice<kRows, 1>(0, j);
    }
    __host__ __device__ ConstSubMatrix<SelfType, 1, kCols> Row(auto i) const
    {
        return Slice<1, kCols>(i, 0);
    }

    __host__ __device__ TransposeView<SelfType> Transpose()
    {
        return TransposeView<SelfType>(*this);
    }
    __host__ __device__ TransposeView<SelfType const> Transpose() const
    {
        return TransposeView<SelfType const>(*this);
    }

    void SetZero() { memset(a, 0, kRows * kCols * sizeof(ScalarType)); }

  private:
    ScalarType* a;
};

template <class /*CMatrix*/ TMatrix, int RepeatRows, int RepeatCols>
class TiledView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = TiledView<NestedType, RepeatRows, RepeatCols>;

    static auto constexpr kRows = RepeatRows * NestedType::kRows;
    static auto constexpr kCols = RepeatCols * NestedType::kCols;

    __host__ __device__ TiledView(NestedType const& A) : A(A) {}

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const
    {
        return A(i % NestedType::kRows, j % NestedType::kCols);
    }

    // Smart accessors
    template <auto S, auto T>
    __host__ __device__ SubMatrix<SelfType, S, T> Slice(auto i, auto j)
    {
        return SubMatrix<SelfType, S, T>(*this, i, j);
    }
    __host__ __device__ SubMatrix<SelfType, kRows, 1> Col(auto j) { return Slice<kRows, 1>(0, j); }
    __host__ __device__ SubMatrix<SelfType, 1, kCols> Row(auto i) { return Slice<1, kCols>(i, 0); }
    __host__ __device__ TransposeView<SelfType> Transpose() const
    {
        return TransposeView<SelfType>(*this);
    }

  private:
    NestedType const& A;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto operator+(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    return Sum<TLhsMatrix, TRhsMatrix>(std::forward<TLhsMatrix>(A), std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto operator+=(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        ((std::forward<TLhsMatrix>(A)(I, j) += std::forward<TRhsMatrix>(B)(I, j)), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<TLhsMatrix::kRows>()), ...);
    };
    fCols(std::make_index_sequence<TLhsMatrix::kCols>());
    return A;
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator-(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    using ScalarType = typename TMatrix::ScalarType;
    return Scale<TMatrix>(ScalarType(-1.), std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto operator-(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    using NegatedMatrixType = Scale<TRhsMatrix>;
    NegatedMatrixType negB  = -std::forward<TRhsMatrix>(B);
    return Sum<TLhsMatrix, NegatedMatrixType>(std::forward<TLhsMatrix>(A), negB);
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto operator-=(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    static_assert(
        LhsMatrixType::kRows == RhsMatrixType::kRows and
            LhsMatrixType::kCols == RhsMatrixType::kCols,
        "A and B must have same dimensions");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        (std::forward<TLhsMatrix>(A)(I, j) -= std::forward<TRhsMatrix>(B)(I, j), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<LhsMatrixType::kRows>()), ...);
    };
    fCols(std::make_index_sequence<LhsMatrixType::kCols>());
    return A;
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto operator*(TLhsMatrix const& A, TRhsMatrix const& B)
{
    return Product<TLhsMatrix, TRhsMatrix>(A, B);
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator*(typename TMatrix::ScalarType k, TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return Scale<TMatrix>(k, std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator*=(TMatrix&& A, typename TMatrix::ScalarType k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        (std::forward<TMatrix>(A)(I, j) *= k, ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<TMatrix::kRows>()), ...);
    };
    fCols(std::make_index_sequence<TMatrix::kCols>());
    return A;
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator/(TMatrix&& A, typename TMatrix::ScalarType k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    using ScalarType = typename MatrixType::ScalarType;
    return Scale<TMatrix>(ScalarType(1. / k), std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator/=(TMatrix&& A, typename TMatrix::ScalarType k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        (std::forward<TMatrix>(A)(I, j) /= k, ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<TMatrix::kRows>()), ...);
    };
    fCols(std::make_index_sequence<TMatrix::kCols>());
    return A;
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto Trace(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    static_assert(TMatrix::kRows == TMatrix::kCols, "Cannot compute trace of non-square matrix");
    auto sum = [&]<auto... I>(std::index_sequence<I...>) {
        return (std::forward<TMatrix>(A)(I, I) + ...);
    };
    return sum(std::make_index_sequence<TMatrix::kRows>{});
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto SquaredNorm(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return Trace(std::forward<TMatrix>(A).Transpose() * std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto Norm(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    using ScalarType = typename MatrixType::ScalarType;
    if constexpr (std::is_same_v<ScalarType, float>)
    {
        return sqrtf(SquaredNorm(std::forward<TMatrix>(A)));
    }
    else
    {
        return sqrt(SquaredNorm(std::forward<TMatrix>(A)));
    }
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto Cross(TLhsMatrix&& A, TRhsMatrix&& B)
{
    return CrossProduct<TLhsMatrix, TRhsMatrix>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto Determinant(TMatrix const& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    static_assert(
        TMatrix::kRows == TMatrix::kCols,
        "Cannot compute determinant of non-square matrix");
    static_assert(TMatrix::kRows < 4, "Determinant of matrix of dimensions >= 4 too costly");
    if constexpr (TMatrix::kRows == 1)
    {
        return std::forward<TMatrix>(A)(0, 0);
    }
    if constexpr (TMatrix::kRows == 2)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }
    if constexpr (TMatrix::kRows == 3)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
               A(0, 1) * (A(1, 0) * A(2, 2) - A(2, 0) * A(1, 2)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1));
    }
    using ScalarType = typename TMatrix::ScalarType;
    return ScalarType{0.};
}

template <auto RepeatRows, auto RepeatCols, class /*CMatrix*/ TMatrix>
__host__ __device__ auto Repeat(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return TiledView<TMatrix, RepeatRows, RepeatCols>(std::forward<TMatrix>(A));
}

} // namespace linalg
} // namespace math
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_MATH_LINALG_MATRIX_CUH