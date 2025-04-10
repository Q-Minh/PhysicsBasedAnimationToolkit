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
    VectorX Fk;   ///< `3|# verts| x 1` vector of current residual
    VectorX Fkm1; ///< `3|# verts| x 1` vector of past residual
    VectorX Gkm1; ///< `3|# verts| x 1` vector of past iterate
    VectorX xkm1; ///< `3|# verts|` past iterate vector
    MatrixX DFK; ///< `3|# verts| x m` matrix of past residuals window used in Anderson acceleration
    MatrixX DGK; ///< `3|# verts| x m` matrix of past iterates window used in Anderson acceleration
    VectorX alpha; ///< `m` vector of Anderson coefficients
    Index mkt;     ///< Past time step's largest window size
    MatrixX DFKt;  ///< `3|# verts| x m` matrix of past residuals window used in Anderson
                   ///< acceleration in beginning of past time step
    MatrixX DGKt; ///< `3|# verts| x m` matrix of past iterates window used in Anderson acceleration
                  ///< in beginning of past time step
    bool mWarmStartAvailable; ///< Whether the warm start is available
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_ACCELERATEDANDERSONINTEGRATOR_H
