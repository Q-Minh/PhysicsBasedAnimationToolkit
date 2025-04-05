/**
 * @file AcceleratedAndersonIntegrator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Accelerated Anderson accelerated VBD integrator
 * @date 2025-04-05
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_VBD_ACCELERATEDANDERSONINTEGRATOR_H
#define PBAT_SIM_VBD_ACCELERATEDANDERSONINTEGRATOR_H

#include "Integrator.h"

namespace pbat::sim::vbd {

/**
 * @brief Accelerated Anderson accelerated VBD integrator
 */
class AcceleratedAndersonIntegrator : public Integrator
{
  public:
    /**
     * @brief Construct a new Accelerated Anderson Integrator object
     *
     * @param data Simulation data
     */
    AcceleratedAndersonIntegrator(Data data);

  protected:
    /**
     * @brief Run the VBD accelerated optimizer on the BDF1 problem
     *
     * @param sdt Substep
     * @param sdt2 Substep squared
     * @param iterations Number of iterations
     */
    virtual void Solve(Scalar sdt, Scalar sdt2, Index iterations) override;

  private:
    VectorX x0;   ///< `3|# verts|` initial iterate vector
    VectorX xkm1; ///< `3|# verts| x 1` vector of past iterate
    VectorX F0;   ///< `3|# verts| x 1` vector of past residual
    VectorX Fk;   ///< `3|# verts| x 1` vector of current residual
    MatrixX DFK; ///< `3|# verts| x m` matrix of past residuals window used in Anderson acceleration
    MatrixX DGK; ///< `3|# verts| x m` matrix of past iterates window used in Anderson acceleration
    VectorX alpha; ///< `m` vector of Anderson coefficients
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_ACCELERATEDANDERSONINTEGRATOR_H
