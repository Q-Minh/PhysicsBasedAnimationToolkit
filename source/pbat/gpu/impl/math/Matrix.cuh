#ifndef PBAT_GPU_IMPL_MATH_MATRIX_H
#define PBAT_GPU_IMPL_MATH_MATRIX_H

#include "pbat/gpu/impl/common/Buffer.cuh"

#include <concepts>
#include <cublas_v2.h>
#include <exception>
#include <type_traits>

namespace pbat::gpu::impl::math {

template <class TMatrix>
concept CMatrix = requires(TMatrix a)
{
    requires std::is_same_v<typename TMatrix::ValueType, float> or
        std::is_same_v<typename TMatrix::ValueType, double>;
    {a.Raw()}->std::same_as<typename TMatrix::ValueType*>;
    {a.Rows()}->std::convertible_to<int>;
    {a.Cols()}->std::convertible_to<int>;
    {a.LeadingDimensions()}->std::convertible_to<int>;
    {a.Operation()}->std::convertible_to<cublasOperation_t>;
};

template <class TVector>
concept CVector = requires(TVector a)
{
    requires std::is_same_v<typename TVector::ValueType, float> or
        std::is_same_v<typename TVector::ValueType, double>;
    {a.Raw()}->std::same_as<typename TVector::ValueType*>;
    {a.Rows()}->std::convertible_to<int>;
    {a.Increment()}->std::convertible_to<int>;
};

template <class T>
struct VectorView
{
    using ValueType = std::remove_cvref_t<T>;

    ValueType* data; ///< Pointer to the vector coefficients
    int n;           ///< Number of rows
    int inc;         ///< Increment

    // Storage information
    ValueType* Raw() { return data; }
    ValueType const* Raw() const { return data; }
    auto Rows() const { return n; }
    constexpr auto Cols() const { return 1; }
    auto Increment() const { return inc; }

    // Accessors
    VectorView<ValueType> Slice(auto row, auto rows, auto inc) const
    {
        return VectorView<ValueType>{const_cast<ValueType*>(data) + row, rows, inc};
    }
    VectorView<ValueType> Segment(auto row, auto rows) const { return Slice(row, rows, 1); }
    VectorView<ValueType> Head(auto rows) const { return Slice(0, rows, 1); }
    VectorView<ValueType> Tail(auto rows) const { return Slice(n - rows, rows, 1); }
};

template <class T>
struct MatrixView
{
    using ValueType = std::remove_cvref_t<T>;

    ValueType* data;      ///< Pointer to the matrix coefficients
    int m;                ///< Number of rows
    int n;                ///< Number of columns
    int ld;               ///< Leading dimension
    cublasOperation_t op; ///< CUBLAS operation type

    MatrixView(
        ValueType* dataIn,
        int mIn,
        int nIn,
        int ldIn,
        cublasOperation_t opIn = cublasOperation_t::CUBLAS_OP_N)
        : data(dataIn), m(mIn), n(nIn), ld(ldIn), op(opIn)
    {
        if (ld < m)
        {
            throw std::invalid_argument(
                "MatrixView::MatrixView(ValueType* data, int m, int n, int ld) -> ld < m");
        }
    }

    template <CVector TVector>
    MatrixView(TVector const& v) : data(v.Raw()), m(v.Rows()), n(1), ld(v.Rows()), op(CUBLAS_OP_N)
    {
        if (v.Increment() != 1)
        {
            throw std::invalid_argument(
                "MatrixView::MatrixView(TVector const& v) -> v.Increment() must be 1");
        }
    }

    // Storage information
    ValueType* Raw() { return data; }
    ValueType const* Raw() const { return data; }
    auto Rows() const { return m; }
    auto Cols() const { return n; }
    auto LeadingDimensions() const { return ld; }
    auto Operation() const { return op; }

    // Accessors
    MatrixView<ValueType> Transposed() const
    {
        return MatrixView<ValueType>{
            const_cast<ValueType*>(data),
            m,
            n,
            ld,
            op == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N};
    }
    MatrixView<ValueType> SubMatrix(auto row, auto col, auto rows, auto cols) const
    {
        return MatrixView<ValueType>{const_cast<ValueType*>(data) + ld * col + row, rows, cols, ld};
    }
    MatrixView<ValueType> LeftCols(auto cols) const { return SubMatrix(0, 0, m, cols); }
    MatrixView<ValueType> RightCols(auto cols) const
    {
        return SubMatrix(0, Cols() - cols, m, cols);
    }
    MatrixView<ValueType> TopRows(auto rows) const { return SubMatrix(0, 0, rows, Cols()); }
    MatrixView<ValueType> BottomRows(auto rows) const
    {
        return SubMatrix(Rows() - rows, 0, rows, Cols());
    }
    MatrixView<ValueType> Col(auto col) const { return SubMatrix(0, col, Rows(), Cols()); }
    MatrixView<ValueType> Row(auto row) const { return SubMatrix(row, 0, 1, Cols()); }
    VectorView<ValueType> Flattened() const
    {
        return VectorView<ValueType>{const_cast<ValueType*>(data), Rows() * Cols(), 1};
    }
};

template <class T>
struct Matrix
{
    using ValueType = std::remove_cvref_t<T>;

    Matrix() = default;
    Matrix(auto rows, auto cols) : data(rows * cols), m(rows) {}

    common::Buffer<ValueType> data; ///< `m x n` dense matrix coefficients in column-major order
    int m;                          ///< Number of rows

    // Storage information
    ValueType* Raw() { return data.Raw(); }
    ValueType const* Raw() const { return data.Raw(); }
    auto Rows() const { return m; }
    auto Cols() const { return static_cast<int>(data.Size()) / m; }
    auto LeadingDimensions() const { return m; }
    auto Operation() const { return CUBLAS_OP_N; }

    // Accessors
    MatrixView<ValueType> View() const
    {
        return MatrixView<ValueType>{const_cast<ValueType*>(data.Raw()), m, Cols(), m};
    }
    MatrixView<ValueType> SubMatrix(auto row, auto col, auto rows, auto cols) const
    {
        ValueType* a = const_cast<ValueType*>(data.Raw());
        return MatrixView<ValueType>{a + m * col + row, rows, cols, m};
    }
    MatrixView<ValueType> LeftCols(auto cols) const { return SubMatrix(0, 0, m, cols); }
    MatrixView<ValueType> RightCols(auto cols) const
    {
        return SubMatrix(0, Cols() - cols, m, cols);
    }
    MatrixView<ValueType> TopRows(auto rows) const { return SubMatrix(0, 0, rows, Cols()); }
    MatrixView<ValueType> BottomRows(auto rows) const
    {
        return SubMatrix(Rows() - rows, 0, rows, Cols());
    }
    MatrixView<ValueType> Col(auto col) const { return SubMatrix(0, col, Rows(), Cols()); }
    MatrixView<ValueType> Row(auto row) const { return SubMatrix(row, 0, 1, Cols()); }
    VectorView<ValueType> Flattened() const
    {
        return VectorView<ValueType>{const_cast<ValueType*>(data.Raw()), Rows() * Cols(), 1};
    }
};

template <class T>
struct Vector
{
    using ValueType = std::remove_cvref_t<T>;

    Vector() = default;
    Vector(auto rows) : data(rows), n(rows) {}

    common::Buffer<ValueType> data; ///< `n x 1` dense vector coefficients
    int n;                          ///< Number of rows

    // Storage information
    ValueType* Raw() { return data.Raw(); }
    ValueType const* Raw() const { return data.Raw(); }
    auto Rows() const { return n; }
    constexpr auto Cols() const { return 1; }
    constexpr auto Increment() const { return 1; }

    // Accessors
    VectorView<ValueType> Slice(auto row, auto rows, auto inc) const
    {
        ValueType* a = const_cast<ValueType*>(data.Raw());
        return VectorView<ValueType>{a + row, rows, inc};
    }
    VectorView<ValueType> Segment(auto row, auto rows) const { return Slice(row, rows, 1); }
    VectorView<ValueType> Head(auto rows) const { return Slice(0, rows, 1); }
    VectorView<ValueType> Tail(auto rows) const { return Slice(n - rows, rows, 1); }
};

}; // namespace pbat::gpu::impl::math

#endif // PBAT_GPU_IMPL_MATH_MATRIX_H
