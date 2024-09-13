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
        TMatrix::kRows
    } -> std::convertible_to<int>;
    {
        TMatrix::kCols
    } -> std::convertible_to<int>;
    // WARNING: This constraint causes compile errors with nvcc, I don't know why!
    // {
    //     M(0, 0)
    // };
};

template <CMatrix TMatrix>
class TransposeView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = TransposeView<NestedType>;

    static auto constexpr kRows = NestedType::kCols;
    static auto constexpr kCols = NestedType::kRows;

    __host__ __device__ TransposeView(NestedType& A) : A(A) {}

    template <class TOtherMatrix>
    __host__ __device__ SelfType& operator=(TOtherMatrix&& B)
    {
        using OtherMatrixType = std::remove_cvref_t<TOtherMatrix>;
        static_assert(CMatrix<OtherMatrixType>, "B must satisfy CMatrix");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TOtherMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
        return *this;
    }

    __host__ __device__ void SetConstant(ScalarType k)
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = k), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
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

template <CMatrix TMatrix>
class ConstTransposeView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = ConstTransposeView<NestedType>;

    static auto constexpr kRows = NestedType::kCols;
    static auto constexpr kCols = NestedType::kRows;

    __host__ __device__ ConstTransposeView(NestedType const& A) : A(A) {}

    __host__ __device__ constexpr auto Rows() const { return A.Cols(); }
    __host__ __device__ constexpr auto Cols() const { return A.Rows(); }

    __host__ __device__ auto operator()(auto i, auto j) const { return A(j, i); }

    __host__ __device__ NestedType const& Transpose() const { return A; }

  private:
    NestedType const& A;
};

template <CMatrix TMatrix, int M, int N>
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
    __host__ __device__ ConstTransposeView<SelfType const> Transpose() const
    {
        return ConstTransposeView<SelfType const>(*this);
    }

  private:
    NestedType const& A;
    int ib, jb;
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

    __host__ __device__ SubMatrix(NestedType& A, auto ib = 0, auto jb = 0) : A(A), ib(ib), jb(jb)
    {
        static_assert(
            NestedType::kRows >= M and NestedType::kCols >= N and M > 0 and N > 0,
            "Invalid submatrix dimensions");
    }

    template <class /*CMatrix*/ TOtherMatrix>
    __host__ __device__ SelfType& operator=(TOtherMatrix&& B)
    {
        using OtherMatrixType = std::remove_cvref_t<TOtherMatrix>;
        static_assert(CMatrix<OtherMatrixType>, "B must satisfy CMatrix");
        static_assert(
            OtherMatrixType::kRows == kRows and OtherMatrixType::kCols == kCols,
            "Invalid submatrix dimensions");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TOtherMatrix>(B)(I, j)), ...);
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
    __host__ __device__ ConstTransposeView<SelfType const> Transpose() const
    {
        return ConstTransposeView<SelfType const>(*this);
    }

  private:
    NestedType& A;
    int ib, jb;
};

template <class TScalar, int M, int N>
class Ones
{
  public:
    using ScalarType = TScalar;
    using SelfType   = Ones<ScalarType, M, N>;

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return ScalarType{1.}; }

    template <auto S, auto T>
    __host__ __device__ Ones<ScalarType, S, T> Slice(auto i, auto j) const
    {
        return Ones<ScalarType, S, T>();
    }
    __host__ __device__ Ones<ScalarType, kRows, 1> Col(auto j) const
    {
        return Ones<ScalarType, kRows, 1>();
    }
    __host__ __device__ Ones<ScalarType, 1, kCols> Row(auto i) const
    {
        return Ones<ScalarType, 1, kCols>();
    }
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }
};

template <class TScalar, int M, int N>
class Identity
{
  public:
    using ScalarType = TScalar;
    using SelfType   = Ones<ScalarType, M, N>;

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const
    {
        return static_cast<ScalarType>(i == j);
    }

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
    __host__ __device__ ConstTransposeView<SelfType const> Transpose() const
    {
        return ConstTransposeView<SelfType const>(*this);
    }
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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

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

    __host__ __device__ Scale(ScalarType k, NestedType const& A) : k(k), A(A) {}

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return k * A(i, j); }

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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

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

    using ScalarType = typename LhsNestedType::ScalarType;
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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TMatrix>
class Diagonal
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = Diagonal<NestedType>;

    static auto constexpr kRows = NestedType::kRows;
    static auto constexpr kCols = 1;

    __host__ __device__ Diagonal(NestedType const& A) : A(A) {}

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return A(i, i); }

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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

  private:
    NestedType const& A;
};

template <CMatrix TMatrix>
class Square
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = Square<NestedType>;

    static auto constexpr kRows = NestedType::kRows;
    static auto constexpr kCols = NestedType::kCols;

    __host__ __device__ Square(NestedType const& A) : A(A) {}

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return A(i, j) * A(i, j); }

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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

  private:
    NestedType const& A;
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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Minimum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Minimum<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows = LhsNestedType::kRows;
    static auto constexpr kCols = RhsNestedType::kCols;

    __host__ __device__ Minimum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "Invalid matrix minimum dimensions");
    }

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return min(A(i, j), B(i, j)); }

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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class Maximum
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Maximum<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows = LhsNestedType::kRows;
    static auto constexpr kCols = RhsNestedType::kCols;

    __host__ __device__ Maximum(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B)
    {
        static_assert(
            LhsNestedType::kRows == RhsNestedType::kRows and
                LhsNestedType::kCols == RhsNestedType::kCols,
            "Invalid matrix maximum dimensions");
    }

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return max(A(i, j), B(i, j)); }

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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
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

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    template <class /*CMatrix*/ TMatrix>
    __host__ __device__ Matrix(TMatrix&& B) : a()
    {
        using MatrixType = std::remove_cvref_t<TMatrix>;
        static_assert(
            MatrixType::kRows == kRows and MatrixType::kCols == kCols,
            "Invalid matrix assignment dimensions");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
    }

    template <class TMatrix>
    __host__ __device__ SelfType& operator=(TMatrix&& B)
    {
        using OtherMatrixType = std::remove_cvref_t<TMatrix>;
        static_assert(CMatrix<OtherMatrixType>, "B must satisfy CMatrix");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
        return *this;
    }

    __host__ __device__ void SetConstant(ScalarType k)
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = k), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
    }

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
    __host__ __device__ ConstTransposeView<SelfType const> Transpose() const
    {
        return ConstTransposeView<SelfType const>(*this);
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

    static auto constexpr kRows = M;
    static auto constexpr kCols = N;

    __host__ __device__ MatrixView(ScalarType* a) : mA(a) {}

    template <class /*CMatrix*/ TMatrix>
    __host__ __device__ SelfType& operator=(TMatrix&& B)
    {
        using OtherMatrixType = std::remove_cvref_t<TMatrix>;
        static_assert(CMatrix<OtherMatrixType>, "B must satisfy CMatrix");
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = std::forward<TMatrix>(B)(I, j)), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
        return *this;
    }

    __host__ __device__ void SetConstant(ScalarType k)
    {
        auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
            (((*this)(I, j) = k), ...);
        };
        auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
            (fRows(J, std::make_index_sequence<kRows>()), ...);
        };
        fCols(std::make_index_sequence<kCols>());
    }

    __host__ __device__ constexpr auto Rows() const { return kRows; }
    __host__ __device__ constexpr auto Cols() const { return kCols; }

    __host__ __device__ auto operator()(auto i, auto j) const { return mA[j * M + i]; }
    __host__ __device__ auto& operator()(auto i, auto j) { return mA[j * M + i]; }

    // Vector(ized) access
    __host__ __device__ auto operator()(auto i) const { return mA[i]; }
    __host__ __device__ auto& operator()(auto i) { return mA[i]; }

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
    __host__ __device__ ConstTransposeView<SelfType const> Transpose() const
    {
        return ConstTransposeView<SelfType const>(*this);
    }

    void SetZero() { memset(mA, 0, kRows * kCols * sizeof(ScalarType)); }

  private:
    ScalarType* mA;
};

template <CMatrix TMatrix, int RepeatRows, int RepeatCols>
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
    __host__ __device__ ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
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
    return Sum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
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
        (fRows(J, std::make_index_sequence<LhsMatrixType::kRows>()), ...);
    };
    fCols(std::make_index_sequence<LhsMatrixType::kCols>());
    return A;
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator-(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    using ScalarType = typename MatrixType::ScalarType;
    return Scale<MatrixType>(ScalarType(-1.), std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto operator-(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    using NegatedMatrixType = Scale<RhsMatrixType>;
    NegatedMatrixType negB  = -std::forward<TRhsMatrix>(B);
    return Sum<LhsMatrixType, NegatedMatrixType>(std::forward<TLhsMatrix>(A), negB);
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
        ((std::forward<TLhsMatrix>(A)(I, j) -= std::forward<TRhsMatrix>(B)(I, j)), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<LhsMatrixType::kRows>()), ...);
    };
    fCols(std::make_index_sequence<LhsMatrixType::kCols>());
    return A;
}

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
__host__ __device__ auto operator*(TLhsMatrix const& A, TRhsMatrix const& B)
{
    return Product<TLhsMatrix, TRhsMatrix>(A, B);
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator*(typename std::remove_cvref_t<TMatrix>::ScalarType k, TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return Scale<MatrixType>(k, std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator*(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::ScalarType k)
{
    return k * std::forward<TMatrix>(A);
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto
operator*=(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::ScalarType k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        ((std::forward<TMatrix>(A)(I, j) *= k), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<MatrixType::kRows>()), ...);
    };
    fCols(std::make_index_sequence<MatrixType::kCols>());
    return A;
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto operator/(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::ScalarType k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    using ScalarType = typename MatrixType::ScalarType;
    return Scale<MatrixType>(ScalarType(1. / k), std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto
operator/=(TMatrix&& A, typename std::remove_cvref_t<TMatrix>::ScalarType k)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {
        ((std::forward<TMatrix>(A)(I, j) /= k), ...);
    };
    auto fCols = [&]<auto... J>(std::index_sequence<J...>) {
        (fRows(J, std::make_index_sequence<MatrixType::kRows>()), ...);
    };
    fCols(std::make_index_sequence<MatrixType::kCols>());
    return A;
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto Trace(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    static_assert(
        MatrixType::kRows == MatrixType::kCols,
        "Cannot compute trace of non-square matrix");
    auto sum = [&]<auto... I>(std::index_sequence<I...>) {
        return (std::forward<TMatrix>(A)(I, I) + ...);
    };
    return sum(std::make_index_sequence<MatrixType::kRows>{});
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto SquaredNorm(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return Trace(std::forward<TMatrix>(A).Transpose() * std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto Squared(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return Square<MatrixType>(std::forward<TMatrix>(A));
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

template <class /*CMatrix*/ TMatrix>
__host__ __device__ auto Normalized(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    static_assert(
        (MatrixType::kRows == 1) or (MatrixType::kCols == 1),
        "Only vectors can be normalized");
    return std::forward<TMatrix>(A) / Norm(std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto Cross(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    return CrossProduct<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <CMatrix TMatrix>
__host__ __device__ auto Determinant(TMatrix const& A)
{
    using MatrixType = TMatrix;
    static_assert(
        MatrixType::kRows == MatrixType::kCols,
        "Cannot compute determinant of non-square matrix");
    static_assert(MatrixType::kRows < 4, "Determinant of matrix of dimensions >= 4 too costly");
    if constexpr (MatrixType::kRows == 1)
    {
        return A(0, 0);
    }
    if constexpr (MatrixType::kRows == 2)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }
    if constexpr (MatrixType::kRows == 3)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) -
               A(0, 1) * (A(1, 0) * A(2, 2) - A(2, 0) * A(1, 2)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1));
    }
    using ScalarType = typename MatrixType::ScalarType;
    return ScalarType{0.};
}

template <class /*CMatrix*/ TMatrix>
__host__ __device__ Matrix<typename TMatrix::ScalarType, TMatrix::kRows, TMatrix::kCols>
Inverse(TMatrix const& A)
{
    using InputMatrixType = TMatrix;
    static_assert(
        InputMatrixType::kRows < 4 and InputMatrixType::kRows > 1,
        "Cannot compute inverse of large matrix or scalar");
    static_assert(
        InputMatrixType::kRows == InputMatrixType::kCols,
        "Cannot compute inverse of non-square matrix");
    using MatrixType = Matrix<
        typename InputMatrixType::ScalarType,
        InputMatrixType::kRows,
        InputMatrixType::kCols>;
    MatrixType Ainv{};
    if constexpr (MatrixType::kRows == 2)
    {
        auto const a0 = 1.0 / (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
        Ainv(0, 0)    = a0 * A(1, 1);
        Ainv(1, 0)    = -a0 * A(1, 0);
        Ainv(0, 1)    = -a0 * A(0, 1);
        Ainv(1, 1)    = a0 * A(0, 0);
    }
    if constexpr (MatrixType::kRows == 3)
    {
        auto const a0 = A(1, 1) * A(2, 2);
        auto const a1 = A(2, 1) * A(1, 2);
        auto const a2 = A(1, 0) * A(2, 1);
        auto const a3 = A(1, 0) * A(2, 2);
        auto const a4 = A(2, 0) * A(1, 1);
        auto const a5 = 1.0 / (a0 * A(0, 0) - a1 * A(0, 0) + a2 * A(0, 2) - a3 * A(0, 1) -
                               a4 * A(0, 2) + A(2, 0) * A(0, 1) * A(1, 2));
        Ainv(0, 0)    = a5 * (a0 - a1);
        Ainv(1, 0)    = a5 * (-a3 + A(2, 0) * A(1, 2));
        Ainv(2, 0)    = a5 * (a2 - a4);
        Ainv(0, 1)    = a5 * (-A(0, 1) * A(2, 2) + A(2, 1) * A(0, 2));
        Ainv(1, 1)    = a5 * (A(0, 0) * A(2, 2) - A(2, 0) * A(0, 2));
        Ainv(2, 1)    = a5 * (-A(0, 0) * A(2, 1) + A(2, 0) * A(0, 1));
        Ainv(0, 2)    = a5 * (A(0, 1) * A(1, 2) - A(1, 1) * A(0, 2));
        Ainv(1, 2)    = a5 * (-A(0, 0) * A(1, 2) + A(1, 0) * A(0, 2));
        Ainv(2, 2)    = a5 * (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
    }
    return Ainv;
}

template <auto RepeatRows, auto RepeatCols, class /*CMatrix*/ TMatrix>
__host__ __device__ auto Repeat(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    static_assert(CMatrix<MatrixType>, "Input must satisfy concept CMatrix");
    return TiledView<MatrixType, RepeatRows, RepeatCols>(std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto Min(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    return Minimum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
__host__ __device__ auto Max(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    static_assert(CMatrix<LhsMatrixType>, "Input must satisfy concept CMatrix");
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    static_assert(CMatrix<RhsMatrixType>, "Input must satisfy concept CMatrix");
    return Maximum<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

} // namespace linalg
} // namespace math
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_MATH_LINALG_MATRIX_CUH