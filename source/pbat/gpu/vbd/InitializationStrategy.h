#ifndef PBAT_GPU_VBD_INITIALIZATION_STRATEGY_H
#define PBAT_GPU_VBD_INITIALIZATION_STRATEGY_H

namespace pbat {
namespace gpu {
namespace vbd {

enum class EInitializationStrategy { Position, Inertia, KineticEnergyMinimum, AdaptiveVbd, AdaptivePbat };

} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_INITIALIZATION_STRATEGY_H