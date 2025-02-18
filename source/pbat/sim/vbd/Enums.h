#ifndef PBAT_SIM_VBD_ENUMS_H
#define PBAT_SIM_VBD_ENUMS_H

namespace pbat::sim::vbd {

enum class EInitializationStrategy {
    Position,
    Inertia,
    KineticEnergyMinimum,
    AdaptiveVbd,
    AdaptivePbat
};

enum class EAccelerationStrategy {
    None,
    Chebyshev,
    TrustRegion
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_ENUMS_H
