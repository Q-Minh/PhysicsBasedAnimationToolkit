/**
 * @file AndersonIntegrator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Anderson accelerated VBD integrator
 * @date 2025-04-05
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_VBD_ANDERSONINTEGRATOR_H
#define PBAT_SIM_VBD_ANDERSONINTEGRATOR_H

#include "Integrator.h"
#include "PhysicsBasedAnimationToolkitExport.h"

namespace pbat::sim::vbd {

/**
 * @brief Anderson accelerated VBD integrator
 */
class AndersonIntegrator : public Integrator
{
  public:
    /**
     * @brief Construct a new Anderson Integrator object
     *
     * @param data Simulation data
     */
    PBAT_API AndersonIntegrator(Data data);

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
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_ANDERSONINTEGRATOR_H
