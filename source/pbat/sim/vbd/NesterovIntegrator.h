#ifndef PBAT_SIM_VBD_NESTEROVINTEGRATOR_H
#define PBAT_SIM_VBD_NESTEROVINTEGRATOR_H

#include "Integrator.h"
#include "PhysicsBasedAnimationToolkitExport.h"

namespace pbat::sim::vbd {

class NesterovIntegrator : public Integrator
{
  public:
    PBAT_API NesterovIntegrator(Data data);

  protected:
    virtual void Solve(Scalar sdt, Scalar sdt2, Index iterations) override;

  private:
    MatrixX xkm1; ///< `3x|# verts|` \f$ x^{k-1} \f$ used in Nesterov's method
    MatrixX yk;   ///< `3x|# verts|` \f$ y^k \f$ used in Nesterov's method
    Scalar L;     ///< Lipschitz constant for the gradient
    Index start;  ///< The first iteration to start Nesterov's method
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_NESTEROVINTEGRATOR_H
