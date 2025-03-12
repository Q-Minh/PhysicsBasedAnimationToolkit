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
    PBAT_API Integrator(Data data);

    PBAT_API void Step(Scalar dt, Index iterations, Index substeps = Index{1});

    PBAT_API Data data;

    virtual ~Integrator() = default;

  protected:
    PBAT_API void InitializeSolve(Scalar sdt, Scalar sdt2);
    PBAT_API void RunVbdIteration(Scalar sdt, Scalar sdt2);
    PBAT_API virtual void Solve(Scalar sdt, Scalar sdt2, Index iterations);
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_INTEGRATOR_H
