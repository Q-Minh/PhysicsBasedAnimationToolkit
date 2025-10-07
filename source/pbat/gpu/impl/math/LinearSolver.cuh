/**
 * @file LinearSolver.cuh
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Linear solver abstractions over cuSolver
 * @date 2025-04-24
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_IMPL_MATH_LINEARSOLVER_CUH
#define PBAT_GPU_IMPL_MATH_LINEARSOLVER_CUH

#include "Blas.cuh"
#include "Matrix.cuh"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/common/Cuda.cuh"

#include <algorithm>
#include <cstdio>
#include <cuda/api.hpp>
#include <cusolverDn.h>
#include <memory>
#include <type_traits>

#define CUSOLVER_CHECK(err)                                                        \
    {                                                                              \
        cusolverStatus_t err_ = (err);                                             \
        if (err_ != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS)                     \
        {                                                                          \
            std::printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cusolver error");                            \
        }                                                                          \
    }

namespace pbat::gpu::impl::math {

class LinearSolver
{
  public:
    LinearSolver(
        cuda::device_t device =
            common::Device(common::EDeviceSelectionPreference::HighestComputeCapability));

    LinearSolver(LinearSolver const&)            = delete;
    LinearSolver(LinearSolver&&)                 = delete;
    LinearSolver& operator=(LinearSolver const&) = delete;
    LinearSolver& operator=(LinearSolver&&)      = delete;

    cusolverDnHandle_t Handle() const { return mCusolverHandle; }

    template <CMatrix TMatrixA, class TScalar = TMatrixA::ValueType>
    int GeqrfWorkspace(TMatrixA const& A) const;

    template <CMatrix TMatrixQR, CVector TVectorTau, class TScalar = TMatrixQR::ValueType>
    void Geqrf(
        TMatrixQR& QR,
        TVectorTau& tau,
        common::Buffer<TScalar>& workspace,
        std::shared_ptr<cuda::stream_t> stream = nullptr) const;

    template <CMatrix TMatrixQ, CMatrix TMatrixB, class TScalar = TMatrixQ::ValueType>
    int OrmqrWorkspace(TMatrixQ const& Q, TMatrixB const& B, bool bMultiplyFromLeft = true) const;

    template <CMatrix TMatrixQ, CVector TVectorB, class TScalar = TMatrixQ::ValueType>
    int OrmqrWorkspace(TMatrixQ const& Q, TVectorB const& B, bool bMultiplyFromLeft = true) const;

    template <
        CMatrix TMatrixQ,
        CVector TVectorTau,
        CMatrix TMatrixB,
        class TScalar = TMatrixQ::ValueType>
    void Ormqr(
        TMatrixQ const& Q,
        TVectorTau const& tau,
        TMatrixB& B,
        common::Buffer<TScalar>& workspace,
        bool bMultiplyFromLeft                 = true,
        std::shared_ptr<cuda::stream_t> stream = nullptr) const;

    template <
        CMatrix TMatrixQ,
        CVector TVectorTau,
        CVector TVectorB,
        class TScalar = TMatrixQ::ValueType>
    void Ormqr(
        TMatrixQ const& Q,
        TVectorTau const& tau,
        TVectorB& B,
        common::Buffer<TScalar>& workspace,
        bool bMultiplyFromLeft                 = true,
        std::shared_ptr<cuda::stream_t> stream = nullptr) const;

    ~LinearSolver();

  protected:
    void TrySetStream(std::shared_ptr<cuda::stream_t> stream) const;

  private:
    cusolverDnHandle_t mCusolverHandle;
    cuda::device_t mDevice;
};

template <CMatrix TMatrixA, class TScalar>
inline int LinearSolver::GeqrfWorkspace(TMatrixA const& A) const
{
    int workspace{0};
    if constexpr (std::is_same_v<TScalar, float>)
    {
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(
            mCusolverHandle,
            A.Rows(),
            A.Cols(),
            nullptr,
            A.LeadingDimensions(),
            &workspace));
    }
    if constexpr (std::is_same_v<TScalar, double>)
    {
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(
            mCusolverHandle,
            A.Rows(),
            A.Cols(),
            nullptr,
            A.LeadingDimensions(),
            &workspace));
    }
    return workspace;
}

template <CMatrix TMatrixQR, CVector TVectorTau, class TScalar>
inline void LinearSolver::Geqrf(
    TMatrixQR& QR,
    TVectorTau& tau,
    common::Buffer<TScalar>& workspace,
    std::shared_ptr<cuda::stream_t> stream) const
{
    TrySetStream(stream);
    if constexpr (std::is_same_v<TScalar, float>)
    {
        CUSOLVER_CHECK(cusolverDnSgeqrf(
            mCusolverHandle,
            QR.Rows(),
            QR.Cols(),
            QR.Raw(),
            QR.LeadingDimensions(),
            tau.Raw(),
            workspace.Raw(),
            static_cast<int>(workspace.Size()),
            nullptr));
    }
    if constexpr (std::is_same_v<TScalar, double>)
    {
        CUSOLVER_CHECK(cusolverDnDgeqrf(
            mCusolverHandle,
            QR.Rows(),
            QR.Cols(),
            QR.Raw(),
            QR.LeadingDimensions(),
            tau.Raw(),
            workspace.Raw(),
            static_cast<int>(workspace.Size()),
            nullptr));
    }
}

template <CMatrix TMatrixQ, CMatrix TMatrixB, class TScalar>
int LinearSolver::OrmqrWorkspace(TMatrixQ const& Q, TMatrixB const& B, bool bMultiplyFromLeft) const
{
    int workspace{0};
    auto side = bMultiplyFromLeft ? cublasSideMode_t::CUBLAS_SIDE_LEFT :
                                    cublasSideMode_t::CUBLAS_SIDE_RIGHT;
    if constexpr (std::is_same_v<TScalar, float>)
    {
        CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(
            mCusolverHandle,
            side,
            Q.Operation(),
            B.Rows(),
            B.Cols(),
            Q.Cols(),
            nullptr,
            Q.LeadingDimensions(),
            nullptr,
            nullptr,
            B.LeadingDimensions(),
            &workspace));
    }
    if constexpr (std::is_same_v<TScalar, double>)
    {
        CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(
            mCusolverHandle,
            side,
            Q.Operation(),
            B.Rows(),
            B.Cols(),
            Q.Cols(),
            nullptr,
            Q.LeadingDimensions(),
            nullptr,
            nullptr,
            B.LeadingDimensions(),
            &workspace));
    }
    return workspace;
}

template <CMatrix TMatrixQ, CVector TVectorB, class TScalar>
inline int
LinearSolver::OrmqrWorkspace(TMatrixQ const& Q, TVectorB const& B, bool bMultiplyFromLeft) const
{
    MatrixView<TScalar> BB(const_cast<TVectorB&>(B));
    return OrmqrWorkspace(Q, BB, bMultiplyFromLeft);
}

template <CMatrix TMatrixQ, CVector TVectorTau, CMatrix TMatrixB, class TScalar>
inline void LinearSolver::Ormqr(
    TMatrixQ const& Q,
    TVectorTau const& tau,
    TMatrixB& B,
    common::Buffer<TScalar>& workspace,
    bool bMultiplyFromLeft,
    std::shared_ptr<cuda::stream_t> stream) const
{
    TrySetStream(stream);
    auto side = bMultiplyFromLeft ? cublasSideMode_t::CUBLAS_SIDE_LEFT :
                                    cublasSideMode_t::CUBLAS_SIDE_RIGHT;
    if constexpr (std::is_same_v<TScalar, float>)
    {
        CUSOLVER_CHECK(cusolverDnSormqr(
            mCusolverHandle,
            side,
            Q.Operation(),
            B.Rows(),
            B.Cols(),
            Q.Cols(),
            Q.Raw(),
            Q.LeadingDimensions(),
            tau.Raw(),
            B.Raw(),
            B.LeadingDimensions(),
            workspace.Raw(),
            static_cast<int>(workspace.Size()),
            nullptr));
    }
    if constexpr (std::is_same_v<TScalar, double>)
    {
        CUSOLVER_CHECK(cusolverDnDormqr(
            mCusolverHandle,
            side,
            Q.Operation(),
            B.Rows(),
            B.Cols(),
            Q.Cols(),
            Q.Raw(),
            Q.LeadingDimensions(),
            tau.Raw(),
            B.Raw(),
            B.LeadingDimensions(),
            workspace.Raw(),
            static_cast<int>(workspace.Size()),
            nullptr));
    }
}

template <CMatrix TMatrixQ, CVector TVectorTau, CVector TVectorB, class TScalar>
inline void LinearSolver::Ormqr(
    TMatrixQ const& Q,
    TVectorTau const& tau,
    TVectorB& B,
    common::Buffer<TScalar>& workspace,
    bool bMultiplyFromLeft,
    std::shared_ptr<cuda::stream_t> stream) const
{
    MatrixView<TScalar> BB(B);
    Ormqr(Q, tau, BB, workspace, bMultiplyFromLeft, stream);
}

} // namespace pbat::gpu::impl::math

#endif // PBAT_GPU_IMPL_MATH_LINEARSOLVER_CUH
