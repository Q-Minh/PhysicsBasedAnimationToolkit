/**
 * @file Enums.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Enums for VBD API.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_ENUMS_H
#define PBAT_SIM_ALGORITHM_VBD_ENUMS_H

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Initialization strategies for the VBD time step minimization
 */
enum class EInitializationStrategy {
    Position,             ///< \f$ x_0 = x(t) \f$
    Inertia,              ///< \f$ x_0 = x(t) + h v(t) \f$
    KineticEnergyMinimum, ///< \f$ x_0 = x(t) + h v(t) + h^2 M^{-1} f_\text{ext} \f$
    AdaptiveVbd,          ///< Adaptive VBD initialization strategy
    AdaptivePbat          ///< Adaptive PBAT initialization strategy
};

/**
 * @brief Solver for the Broyden least-squares problem
 */
enum class EBroydenLeastSquaresSolver {
    QR,                    ///< QR decomposition
    COD,                   ///< Complete orthogonal decomposition
    LSCG,                  ///< Conjugate gradient
    OneStepSteepestDescent ///< One-step steepest descent
};

/**
 * @brief Broyden Jacobian estimate strategies
 */
// clang-format off
enum class EBroydenJacobianEstimate {
    Identity, ///< Initial Jacobian is identity matrix
    ScaledIdentity, ///< See \cite oren1974SelfScaling
    QuasiCauchyRelationDiagonalUpdating, ///< See \cite zhu1999quasi
    UsdDiagonal, ///< See \cite marjugi2013diagonal
    DiagonalCauchySchwarz ///< Ours
};
// clang-format on

/**
 * @brief Homogenization strategy
 */
enum class EHomogenizationStrategy {
    None,                                                      ///< No homogenization
    HomogeneousElasticityWithDynamicsMatchingContactStiffness, ///< Homogenize elastic material and
                                                               ///< ensure dynamics matching contact
                                                               ///< stiffness in the spirit of \cite
                                                               ///< ando_cubic_2024
    Conditioning, ///< Conditioning homogenization
};

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_ENUMS_H
