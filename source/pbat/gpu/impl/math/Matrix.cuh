#ifndef PBAT_GPU_IMPL_MATH_MATRIX_H
#define PBAT_GPU_IMPL_MATH_MATRIX_H

#include "pbat/gpu/impl/common/Buffer.cuh"

#include <concepts>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda/api/stream.hpp>
#include <exception>
#include <type_traits>

#define CUBLAS_CHECK(err)                                                        \
    {                                                                            \
        cublasStatus_t err_ = (err);                                             \
        if (err_ != CUBLAS_STATUS_SUCCESS)                                       \
        {                                                                        \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                            \
        }                                                                        \
    }

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
};

template <class TVector>
concept CVector = requires(TVector a)
{
    requires std::is_same_v<typename TVector::ValueType, float> or
        std::is_same_v<typename TVector::ValueType, double>;
    {a.Raw()}->std::same_as<typename TVector::ValueType*>;
    {a.Rows()}->std::convertible_to<int>;
    {a.Cols()}->std::convertible_to<int>;
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

class Blas
{
  public:
    Blas();
    Blas(Blas const&)            = delete;
    Blas(Blas&&)                 = delete;
    Blas& operator=(Blas const&) = delete;
    Blas& operator=(Blas&&)      = delete;

    template <
        CMatrix TMatrixA,
        CVector TVectorX,
        CVector TVectorY,
        class TScalar = TMatrixA::ValueType>
    void Gemv(
        TMatrixA const& A,
        TVectorX const& x,
        TVectorY& y,
        TScalar alpha                = 1.0,
        TScalar beta                 = 0.0,
        cuda::stream_t const& stream = cuda::device::current::get().default_stream())
    {
        stream.device().make_current();
        CUBLAS_CHECK(cublasSetStream(mHandle, stream.handle()));
        if constexpr (std::is_same_v<TScalar, float>)
        {
            CUBLAS_CHECK(cublasSgemv(
                mHandle,
                A.Operation(),
                A.Rows(),
                A.Cols(),
                &alpha,
                A.Raw(),
                A.LeadingDimensions(),
                x.Raw(),
                x.Increment(),
                &beta,
                y.Raw(),
                y.Increment()));
        }
        if constexpr (std::is_same_v<TScalar, double>)
        {
            CUBLAS_CHECK(cublasDgemv(
                mHandle,
                A.Operation(),
                A.Rows(),
                A.Cols(),
                &alpha,
                A.Raw(),
                A.LeadingDimensions(),
                x.Raw(),
                x.Increment(),
                &beta,
                y.Raw(),
                y.Increment()));
        }
    }

    ~Blas();

  private:
    cublasHandle_t mHandle; ///< CUBLAS handle
};

}; // namespace pbat::gpu::impl::math

#endif // PBAT_GPU_IMPL_MATH_MATRIX_H
