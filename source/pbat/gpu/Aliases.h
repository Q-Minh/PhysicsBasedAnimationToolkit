#ifndef PBAT_GPU_ALIASES_H
#define PBAT_GPU_ALIASES_H

#include <Eigen/Core>
#include <cstdint>

namespace pbat {

using GpuScalar = float;
using GpuIndex  = std::int32_t;

using GpuMatrixX      = Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic>;
using GpuIndexMatrixX = Eigen::Matrix<GpuIndex, Eigen::Dynamic, Eigen::Dynamic>;

using GpuVectorX      = Eigen::Vector<GpuScalar, Eigen::Dynamic>;
using GpuIndexVectorX = Eigen::Vector<GpuIndex, Eigen::Dynamic>;

} // namespace pbat

#endif // PBAT_GPU_ALIASES_H
