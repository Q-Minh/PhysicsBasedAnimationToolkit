#ifndef PBAT_SIM_VBD_INTEGRATOR_H
#define PBAT_SIM_VBD_INTEGRATOR_H

#include "Data.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

class Integrator
{
  public:
    PBAT_API Integrator(Data&& data);

    PBAT_API void
    Step(Scalar dt, Index iterations, Index substeps = Index{1}, Scalar rho = Scalar{1});

    PBAT_API Data data;
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_INTEGRATOR_H