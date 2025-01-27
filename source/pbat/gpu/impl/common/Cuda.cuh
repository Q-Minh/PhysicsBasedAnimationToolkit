#ifndef PBAT_GPU_IMPL_COMMON_CUDA_H
#define PBAT_GPU_IMPL_COMMON_CUDA_H

#include <cuda/api/device.hpp>
#include <cuda/api/devices.hpp>

namespace pbat {
namespace gpu {
namespace impl {
namespace common {

enum class EDeviceSelectionPreference { Default, HighestComputeCapability };

cuda::device_t Device(EDeviceSelectionPreference preference);

} // namespace common
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_COMMON_CUDA_H
