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

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_ENUMS_H
