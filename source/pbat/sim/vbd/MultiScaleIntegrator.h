#ifndef PBAT_SIM_VBD_MULTI_SCALE_INTEGRATOR_H
#define PBAT_SIM_VBD_MULTI_SCALE_INTEGRATOR_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

class Hierarchy;

class MultiScaleIntegrator
{
  public:
    void Step(Scalar dt, Index substeps, Hierarchy& H);

  private:

};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTI_SCALE_INTEGRATOR_H