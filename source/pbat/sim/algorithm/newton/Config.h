/**
 * @file Config.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for the Newton integrator's configuration.
 * @date 2025-05-05
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_ALGORITHM_NEWTON_CONFIG_H
#define PBAT_SIM_ALGORITHM_NEWTON_CONFIG_H

#include "pbat/Aliases.h"

namespace pbat::sim::algorithm::newton {

/**
 * @brief Newton integrator configuration
 */
struct Config
{
    int nSubsteps{1}; ///< Number of substeps for the Newton integrator

    int nMaxIterations{10};              ///< Maximum number of iterations for the Newton integrator
    Scalar gtol{1e-4};                   ///< Gradient norm threshold for convergence
    int nMaxLinearSolverIterations{150}; ///< Maximum number of iterations for the linear solver
    Scalar rtol{1e-6};                   ///< Relative tolerance for the linear solver

    int nMaxLineSearchIterations{20}; ///< Maximum number of iterations for the line search
    Scalar tauArmijo{0.5};            ///< Armijo step size decrease factor
    Scalar cArmijo{1e-4};             ///< Armijo slope scale

    Scalar muC{1e6}; ///< Uniform collision penalty

    /**
     * @brief Set the number of substeps for the Newton integrator
     * @param substeps Number of substeps
     * @return Reference to this configuration
     */
    Config& WithSubsteps(int substeps);
    /**
     * @brief Set the convergence parameters for the Newton integrator
     * @param maxIterations Maximum number of iterations
     * @param gtol Gradient norm threshold for convergence
     * @param maxLinearSolverIterations Maximum number of iterations for the linear solver
     * @param rtol Relative tolerance for the linear solver
     * @return Reference to this configuration
     */
    Config&
    WithConvergence(int maxIterations, Scalar gtol, int maxLinearSolverIterations, Scalar rtol);
    /**
     * @brief Set the line search parameters for the Newton integrator
     * @param maxLineSearchIterations Maximum number of iterations for the line search
     * @param tauArmijo Armijo step size decrease factor
     * @param cArmijo Armijo slope scale
     * @return Reference to this configuration
     */
    Config& WithLineSearch(int maxLineSearchIterations, Scalar tauArmijo, Scalar cArmijo);
    /**
     * @brief Set the contact parameters for the Newton integrator
     * @param muC Uniform collision penalty
     * @return Reference to this configuration
     */
    Config& WithContactParameters(Scalar muC);
    /**
     * @brief Finalize construction of the configuration
     * @return Config object
     * @throws std::invalid_argument if any parameter is invalid
     */
    Config& Construct();
};

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_CONFIG_H
