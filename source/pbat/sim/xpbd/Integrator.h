#ifndef PBAT_SIM_XPBD_INTEGRATOR_H
#define PBAT_SIM_XPBD_INTEGRATOR_H

#include "Data.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/geometry/TriangleAabbHierarchy.h"

namespace pbat {
namespace sim {
namespace xpbd {

class Integrator
{
  public:
    PBAT_API Integrator(Data data);

    PBAT_API void Step(Scalar dt, Index iterations, Index substeps = Index{1});

    PBAT_API Data data;

  protected:
    void ProjectBlockNeoHookeanConstraints(Scalar dt, Scalar dt2);
    void ProjectClusteredBlockNeoHookeanConstraints(Scalar dt, Scalar dt2);
    void ProjectBlockNeoHookeanConstraint(Index c, Scalar dt, Scalar dt2);

  private:
    geometry::TetrahedralAabbHierarchy mTetrahedralBvh;
    geometry::TriangleAabbHierarchy<3> mTriangleBvh;
    std::vector<Index> mParticlesInContact;
};

} // namespace xpbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_XPBD_INTEGRATOR_H