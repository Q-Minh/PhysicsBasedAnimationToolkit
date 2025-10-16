#ifndef PBAT_SIM_VBD_ENUMS_H
#define PBAT_SIM_VBD_ENUMS_H

namespace pbat::sim::vbd {

/**
 * @brief Initialization strategies for the VBD time step minimization
 */
enum class EInitializationStrategy {
    Position,
    Inertia,
    KineticEnergyMinimum,
    AdaptiveVbd,
    AdaptivePbat
};

/**
 * @brief Acceleration strategies for the VBD time step minimization
 */
// clang-format off
enum class EAccelerationStrategy {
    None,
    Chebyshev,
    Anderson,
    Nesterov,
    Broyden
};
// clang-format on

/**
 * @brief Broyden Jacobian estimate strategies
 */
// clang-format off
enum class EBroydenJacobianEstimate {
    Identity, ///< Initial Jacobian is identity matrix
    ScaledIdentity, ///< @cite oren1974SelfScaling
    QuasiCauchyRelationDiagonalUpdating, ///< @cite zhu1999quasi
    UsdDiagonal, ///< @cite marjugi2013diagonal
    DiagonalCauchySchwarz ///< Ours
};
// clang-format on

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_ENUMS_H
