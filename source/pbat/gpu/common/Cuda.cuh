#ifndef PBAT_GPU_COMMON_CUDA_CUH
#define PBAT_GPU_COMMON_CUDA_CUH

#include <cuda/api/device.hpp>
#include <cuda/api/devices.hpp>

namespace pbat {
namespace gpu {
namespace common {

enum class EDeviceSelectionPreference
{
    Default,
    HighestComputeCapability
};

cuda::device_t Device(EDeviceSelectionPreference preference);

} // namespace common
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_COMMON_CUDA_CUH