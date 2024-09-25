// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Cuda.cuh"

#include <algorithm>

namespace pbat {
namespace gpu {
namespace common {

cuda::device_t Device(EDeviceSelectionPreference preference)
{
    switch (preference)
    {
        case EDeviceSelectionPreference::HighestComputeCapability: {
            auto deviceIt = std::max_element(
                cuda::devices().begin(),
                cuda::devices().end(),
                [](cuda::device_t d1, cuda::device_t d2) {
                    return d1.compute_capability() < d2.compute_capability();
                });
            return *deviceIt;
        }
        case EDeviceSelectionPreference::Default:
        default: return cuda::device::get(cuda::device::default_device_id);
    }
}

} // namespace common
} // namespace gpu
} // namespace pbat
