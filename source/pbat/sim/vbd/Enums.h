#ifndef PBAT_SIM_VBD_ENUMS_H
#define PBAT_SIM_VBD_ENUMS_H

namespace pbat {
namespace sim {
namespace vbd {

enum class EInitializationStrategy {
    Position,
    Inertia,
    KineticEnergyMinimum,
    AdaptiveVbd,
    AdaptivePbat
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_ENUMS_H