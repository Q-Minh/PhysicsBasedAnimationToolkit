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
    MatrixX FK;  ///< `3|# verts| x m` matrix of past residuals window used in Anderson acceleration
    MatrixX GK;  ///< `3|# verts| x m` matrix of past iterates window used in Anderson acceleration
    MatrixX DFK; ///< `3|# verts| x m` matrix of past residuals window used in Anderson acceleration
    MatrixX DGK; ///< `3|# verts| x m` matrix of past iterates window used in Anderson acceleration
    MatrixX D;   ///< `3|# verts| x m` least-squares matrix
    MatrixX xkm1;  ///< `3x|# verts|` matrix of previous iterate
    VectorX alpha; ///< `m` vector of Anderson coefficients
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_ANDERSONINTEGRATOR_H
