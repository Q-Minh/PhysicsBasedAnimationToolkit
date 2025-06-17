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
    BroydenIntegrator(Data data);

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
    MatrixX U;    ///< `kDims|# verts| x m` matrix of rank-1 update vectors u
    MatrixX V;    ///< `kDims|# verts| x m` matrix of rank-1 update vectors v
    VectorX xkm1; ///< `kDims|# verts|` vector of previous x
    VectorX dx;   ///< `kDims|# verts|` vector of dx = x - xkm1
    VectorX Fk;   ///< `kDims|# verts|` vector of root-finding function
    VectorX Fkm1; ///< `kDims|# verts|` vector of previous root-finding function
    VectorX GdFk; ///< `kDims|# verts|` vector of Gk * dFk
    VectorX GTdx; ///< `kDims|# verts|` vector of Gk^T * dx
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_BROYDENINTEGRATOR_H
