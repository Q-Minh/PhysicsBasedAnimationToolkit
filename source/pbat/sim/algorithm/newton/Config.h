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

    Scalar muC{1e6};   ///< Uniform collision penalty
    Scalar muF{0.3};   ///< Uniform friction coefficient
    Scalar epsv{1e-3}; ///< IPC \cite li2020ipc 's relative velocity threshold for static to dynamic
                       ///< friction's smooth transition

    Config& WithSubsteps(int substeps);
    Config&
    WithConvergence(int maxIterations, Scalar gtol, int maxLinearSolverIterations, Scalar rtol);
    Config& WithLineSearch(int maxLineSearchIterations, Scalar tauArmijo, Scalar cArmijo);
    Config& WithContactParameters(Scalar muC, Scalar muF, Scalar epsv);
    Config& Construct();
};

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_CONFIG_H
