#include "Prolongation.h"

#include "Level.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

Prolongation::Prolongation(VolumeMesh const& FM, VolumeMesh const& CM) : ec(), Nc()
{
    geometry::TetrahedralAabbHierarchy cbvh(CM.X, CM.E);
    ec       = cbvh.PrimitivesContainingPoints(FM.X);
    auto Xig = fem::ReferencePositions(CM, ec, FM.X);
    Nc       = fem::ShapeFunctionsAt<VolumeMesh::ElementType>(Xig);
}

void Prolongation::Apply(Level const& lc, Level& lf)
{
    VolumeMesh const& CM = lc.mesh;
    MatrixX const& xc    = lc.x;
    MatrixX& xf          = lf.x;
    tbb::parallel_for(Index(0), ec.size(), [&](Index i) {
        auto e    = ec(i);
        auto inds = CM.E(Eigen::placeholders::all, e);
        auto N    = Nc.col(i);
        xf.col(i) = xc(Eigen::placeholders::all, inds) * N;
    });
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat