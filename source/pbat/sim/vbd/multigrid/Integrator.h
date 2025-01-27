#ifndef PBAT_SIM_VBD_MULTIGRID_INTEGRATOR_H
#define PBAT_SIM_VBD_MULTIGRID_INTEGRATOR_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Hierarchy;

class Integrator
{
  public:
    void Step(Scalar dt, Index substeps, Hierarchy& hierarchy) const;
    void ComputeAndSortStrainRates(Hierarchy& H, Scalar sdt) const;
    void ComputeInertialTargetPositions(Hierarchy& H, Scalar sdt, Scalar sdt2) const;
    void InitializeBCD(Hierarchy& H, Scalar sdt, Scalar sdt2) const; 
    void UpdateVelocity(Hierarchy& H, Scalar sdt) const; 

  private:
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_INTEGRATOR_H
