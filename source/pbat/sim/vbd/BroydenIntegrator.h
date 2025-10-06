/**
 * @file BroydenIntegrator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Broyden VBD integrator
 * @date 2025-04-05
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_VBD_BROYDENINTEGRATOR_H
#define PBAT_SIM_VBD_BROYDENINTEGRATOR_H

#include "Integrator.h"
#include "PhysicsBasedAnimationToolkitExport.h"

namespace pbat::sim::vbd {

/**
 * @brief Accelerated Anderson accelerated VBD integrator
 */
class BroydenIntegrator : public Integrator
{
  public:
    /**
     * @brief Construct a new Broyden Integrator object
     *
     * @param data Simulation data
     */
    PBAT_API BroydenIntegrator(Data data);

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
    MatrixX GvbdFk;  ///< `|# dofs| x m` vbd-preconditioned gradient differences
    MatrixX Xk;      ///< `|# dofs| x m` past steps
    VectorX gammak;  ///< `m x 1` subspace residual
    VectorX xkm1;    ///< `|# dofs| x 1` previous step
    VectorX vbdfk;   ///< `|# dofs| x 1` vbd step
    VectorX vbdfkm1; ///< `|# dofs| x 1` past vbd step
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_BROYDENINTEGRATOR_H
