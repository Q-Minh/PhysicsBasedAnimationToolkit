#ifndef PBAT_GPU_IMPL_MATH_BLAS_H
#define PBAT_GPU_IMPL_MATH_BLAS_H

#include "Matrix.cuh"
#include "pbat/gpu/impl/common/Cuda.cuh"

#include <cstdio>
#include <cublas_v2.h>
#include <cuda/api/stream.hpp>
#include <exception>
#include <memory>
#include <type_traits>

#define CUBLAS_CHECK(err)                                                        \
    {                                                                            \
        cublasStatus_t err_ = (err);                                             \
        if (err_ != cublasStatus_t::CUBLAS_STATUS_SUCCESS)                       \
        {                                                                        \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                            \
        }                                                                        \
    }

namespace pbat::gpu::impl::math {

class Blas
{
  public:
    Blas(
        cuda::device_t device =
            common::Device(common::EDeviceSelectionPreference::HighestComputeCapability));
    Blas(Blas const&)            = delete;
    Blas(Blas&&)                 = delete;
    Blas& operator=(Blas const&) = delete;
    Blas& operator=(Blas&&)      = delete;

    cublasHandle_t Handle() const { return mHandle; }

    template <
        CMatrix TMatrixA,
        CVector TVectorX,
        CVector TVectorY,
        class TScalar = TMatrixA::ValueType>
    void Gemv(
        TMatrixA const& A,
        TVectorX const& x,
        TVectorY& y,
        TScalar alpha                          = TScalar(1),
        TScalar beta                           = TScalar(0),
        std::shared_ptr<cuda::stream_t> stream = nullptr) const;

    template <CMatrix TMatrixA, CMatrix TMatrixB, class TScalar = TMatrixA::ValueType>
    void UpperTriangularSolve(
        TMatrixA const& A,
        TMatrixB& B,
        TScalar alpha                          = TScalar(1),
        bool bHasUnitDiagonal                  = false,
        std::shared_ptr<cuda::stream_t> stream = nullptr) const;

    template <CMatrix TMatrixA, CMatrix TMatrixB, class TScalar = TMatrixA::ValueType>
    void LowerTriangularSolve(
        TMatrixA const& A,
        TMatrixB& B,
        TScalar alpha                          = TScalar(1),
        bool bHasUnitDiagonal                  = false,
        std::shared_ptr<cuda::stream_t> stream = nullptr) const;

    template <CMatrix TMatrixA, CMatrix TMatrixB, class TScalar = TMatrixA::ValueType>
    void Trsm(
        cublasSideMode_t side,
        cublasFillMode_t uplo,
        cublasDiagType_t diag,
        TMatrixA const& A,
        TMatrixB& B,
        TScalar alpha                          = TScalar(1),
        std::shared_ptr<cuda::stream_t> stream = nullptr) const;

    ~Blas();

  protected:
    void TrySetStream(std::shared_ptr<cuda::stream_t> stream) const;

  private:
    cublasHandle_t mHandle; ///< CUBLAS handle
    cuda::device_t mDevice; ///< Device handle
};

template <CMatrix TMatrixA, CVector TVectorX, CVector TVectorY, class TScalar>
inline void Blas::Gemv(
    TMatrixA const& A,
    TVectorX const& x,
    TVectorY& y,
    TScalar alpha,
    TScalar beta,
    std::shared_ptr<cuda::stream_t> stream) const
{
    TrySetStream(stream);
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

template <CMatrix TMatrixA, CMatrix TMatrixB, class TScalar>
inline void Blas::UpperTriangularSolve(
    TMatrixA const& A,
    TMatrixB& B,
    TScalar alpha,
    bool bHasUnitDiagonal,
    std::shared_ptr<cuda::stream_t> stream) const
{
    Trsm(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_UPPER,
        bHasUnitDiagonal ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
        A,
        B,
        alpha,
        stream);
}

template <CMatrix TMatrixA, CMatrix TMatrixB, class TScalar>
inline void Blas::LowerTriangularSolve(
    TMatrixA const& A,
    TMatrixB& B,
    TScalar alpha,
    bool bHasUnitDiagonal,
    std::shared_ptr<cuda::stream_t> stream) const
{
    Trsm(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        bHasUnitDiagonal ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
        A,
        B,
        alpha,
        stream);
}

template <CMatrix TMatrixA, CMatrix TMatrixB, class TScalar>
inline void Blas::Trsm(
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasDiagType_t diag,
    TMatrixA const& A,
    TMatrixB& B,
    TScalar alpha,
    std::shared_ptr<cuda::stream_t> stream) const
{
    TrySetStream(stream);
    if constexpr (std::is_same_v<TScalar, double>)
    {
        cublasDtrsm(
            mHandle,
            side,
            uplo,
            A.Operation(),
            diag,
            B.Rows(),
            B.Cols(),
            &alpha,
            A.Raw(),
            A.LeadingDimensions(),
            B.Raw(),
            B.LeadingDimensions());
    }
    if constexpr (std::is_same_v<TScalar, float>)
    {
        cublasStrsm(
            mHandle,
            side,
            uplo,
            A.Operation(),
            diag,
            B.Rows(),
            B.Cols(),
            &alpha,
            A.Raw(),
            A.LeadingDimensions(),
            B.Raw(),
            B.LeadingDimensions());
    }
}

} // namespace pbat::gpu::impl::math

#endif // PBAT_GPU_IMPL_MATH_BLAS_H
