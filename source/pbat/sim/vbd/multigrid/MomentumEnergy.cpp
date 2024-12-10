#include "MomentumEnergy.h"

#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

MomentumEnergy::MomentumEnergy(Data const& problem, VolumeMesh const& CM, CageQuadrature const& CQ)
    : xtildeg(), erg(), Nrg(), Ncg(), rhog()
{
    geometry::TetrahedralAabbHierarchy rbvh(problem.mesh.X, problem.mesh.E);
    erg  = rbvh.NearestPrimitivesToPoints(CQ.Xg).first;
    Nrg  = fem::ShapeFunctionsAt(problem.mesh, erg, CQ.Xg);
    Ncg  = fem::ShapeFunctionsAt(CM, CQ.eg, CQ.Xg);
    rhog = problem.rhoe(erg);
    xtildeg.resize(3, rhog.size());
}

void MomentumEnergy::UpdateInertialTargetPositions(Data const& problem)
{
    tbb::parallel_for(Index(0), erg.size(), [&](Index g) {
        auto e         = erg(g);
        auto inds      = problem.mesh.E.col(e);
        auto N         = Nrg.col(g);
        auto xtilde    = problem.xtilde(Eigen::placeholders::all, inds);
        xtildeg.col(g) = xtilde * N;
    });
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat