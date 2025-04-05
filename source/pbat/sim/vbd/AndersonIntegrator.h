#ifndef PBAT_SIM_VBD_ANDERSONINTEGRATOR_H
#define PBAT_SIM_VBD_ANDERSONINTEGRATOR_H

#include "Integrator.h"

namespace pbat::sim::vbd {

class AndersonIntegrator : public Integrator
{
  public:
    AndersonIntegrator(Data data);

  protected:
    virtual void Solve(Scalar sdt, Scalar sdt2, Index iterations) override;

  private:
    VectorX Fk;   ///< `3|# verts| x 1` vector of current residual
    VectorX F0;   ///< `3|# verts| x 1` vector of past residual
    VectorX Gkm1; ///< `3|# verts| x 1` vector of past iterate
    MatrixX DFK; ///< `3|# verts| x m` matrix of past residuals window used in Anderson acceleration
    MatrixX DGK; ///< `3|# verts| x m` matrix of past iterates window used in Anderson acceleration
    VectorX x0;  ///< `3|# verts|` initial iterate vector
    VectorX alpha; ///< `m` vector of Anderson coefficients
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_ANDERSONINTEGRATOR_H
