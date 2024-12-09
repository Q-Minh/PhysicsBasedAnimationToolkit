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
    void Step(Scalar dt, Index substeps, Hierarchy& hierarchy);

  private:
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_INTEGRATOR_H