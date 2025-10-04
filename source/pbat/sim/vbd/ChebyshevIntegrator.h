#ifndef PBAT_SIM_VBD_CHEBYSHEVINTEGRATOR_H
#define PBAT_SIM_VBD_CHEBYSHEVINTEGRATOR_H

#include "Integrator.h"
#include "PhysicsBasedAnimationToolkitExport.h"

namespace pbat::sim::vbd {

class ChebyshevIntegrator : public Integrator
{
  public:
    PBAT_API ChebyshevIntegrator(Data data);

  protected:
    virtual void Solve(Scalar sdt, Scalar sdt2, Index iterations) override;

  private:
    MatrixX xkm1; ///< `3x|# verts|` \f$ x^{k-1} \f$ used in Chebyshev semi-iterative method
    MatrixX xkm2; ///< `3x|# verts|` \f$ x^{k-2} \f$ used in Chebyshev semi-iterative method
};

} // namespace pbat::sim::vbd

#endif // PBAT_SIM_VBD_CHEBYSHEVINTEGRATOR_H
