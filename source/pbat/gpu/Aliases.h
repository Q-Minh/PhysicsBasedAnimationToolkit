#ifndef PBAT_GPU_ALIASES_H
#define PBAT_GPU_ALIASES_H

namespace pbat {

using GpuScalar = float;
using GpuIndex  = int;

using GpuMatrixX      = Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic>;
using GpuIndexMatrixX = Eigen::Matrix<GpuIndex, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace pbat

#endif // PBAT_GPU_ALIASES_H