#ifndef PBAT_GPU_COMMON_EIGEN_CUH
#define PBAT_GPU_COMMON_EIGEN_CUH

#include "Buffer.cuh"
#include "pbat/common/Eigen.h"

#include <Eigen/Core>

namespace pbat {
namespace gpu {
namespace common {

template <
    class ScalarType,
    auto D,
    auto StorageOrder = Eigen::ColMajor,
    auto EigenRows    = Eigen::Dynamic,
    auto EigenCols    = Eigen::Dynamic>
void ToBuffer(
    Eigen::Ref<Eigen::Matrix<ScalarType, EigenRows, EigenCols, StorageOrder> const> const& A,
    Buffer<ScalarType, D>& buf)
{
    using SizeType = decltype(buf.Size());
    if constexpr (D > 1)
    {
        if (static_cast<SizeType>(A.rows()) != buf.Dimensions() and
            static_cast<SizeType>(A.cols()) != buf.Size())
        {
            std::ostringstream ss{};
            ss << "Expected input dimensions " << buf.Dimensions() << "x" << buf.Size()
               << ", but got " << A.rows() << "x" << A.cols() << "\n";
            throw std::invalid_argument(ss.str());
        }
        for (auto d = 0; d < buf.Dimensions(); ++d)
        {
            thrust::copy(A.row(d).begin(), A.row(d).end(), buf[d].begin());
        }
    }
    else
    {
        if (static_cast<SizeType>(A.size()) != buf.Size())
        {
            std::ostringstream ss{};
            ss << "Expected input dimensions " << buf.Dimensions() << "x" << buf.Size()
               << " or its transpose, but got " << A.rows() << "x" << A.cols() << "\n";
            throw std::invalid_argument(ss.str());
        }
        thrust::copy(A.data(), A.data() + A.size(), buf.Data());
    }
}

template <class ScalarType, auto D, auto StorageOrder = (D > 1) ? Eigen::RowMajor : Eigen::ColMajor>
Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder>
ToEigen(Buffer<ScalarType, D> const& buf)
{
    if constexpr (D > 1)
    {
        return pbat::common::ToEigen(buf.Get()).reshaped(buf.Size(), buf.Dimensions()).transpose();
    }
    else
    {
        return pbat::common::ToEigen(buf.Get());
    }
}

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_EIGEN_CUH