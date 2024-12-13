#ifndef PBAT_SIM_VBD_LOD_INTEGRATOR_H
#define PBAT_SIM_VBD_LOD_INTEGRATOR_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

struct Hierarchy;

class Integrator
{
  public:
    void Step(Scalar dt, Index substeps, Hierarchy& hierarchy) const;

  private:
};

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LOD_INTEGRATOR_H