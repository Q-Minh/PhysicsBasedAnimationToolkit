#include "ElasticEnergy.h"

#include "pbat/geometry/TetrahedralAabbHierarchy.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

ElasticEnergy::ElasticEnergy(Data const& problem, CageQuadrature const& CQ) : mug(), lambdag()
{
    geometry::TetrahedralAabbHierarchy bvh(problem.mesh.X, problem.mesh.E);
    IndexVectorX const efg = bvh.NearestPrimitivesToPoints(CQ.Xg).first;
    mug                    = problem.lame(0, efg);
    lambdag                = problem.lame(1, efg);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat
