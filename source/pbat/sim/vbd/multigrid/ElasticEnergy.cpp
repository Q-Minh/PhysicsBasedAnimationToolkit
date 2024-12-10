#include "ElasticEnergy.h"

#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

ElasticEnergy::ElasticEnergy(Data const& problem, VolumeMesh const& CM, CageQuadrature const& CQ)
    : mug(), lambdag(), GNg()
{
    geometry::TetrahedralAabbHierarchy bvh(problem.mesh.X, problem.mesh.E);
    IndexVectorX const efg = bvh.NearestPrimitivesToPoints(CQ.Xg).first;
    mug                    = problem.lame(0, efg);
    lambdag                = problem.lame(1, efg);
    GNg                    = fem::ShapeFunctionGradientsAt(CM, efg, CQ.Xg);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat
