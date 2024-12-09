#include "Restriction.h"

#include "Kernels.h"
#include "Level.h"
#include "Quadrature.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <cmath>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

Restriction::Restriction(
    Data const& problem,
    VolumeMesh const& FM,
    VolumeMesh const& CM,
    CageQuadrature const& CQ)
    : efg(), Nfg(), xfg(), rhog(), mug(), lambdag(), Ncg(), GNcg()
{
    geometry::TetrahedralAabbHierarchy fbvh(FM.X, FM.E);
    efg              = fbvh.NearestPrimitivesToPoints(CQ.Xg).first;
    auto fXig        = fem::ReferencePositions(FM, efg, CQ.Xg);
    Nfg              = fem::ShapeFunctionsAt<VolumeMesh::ElementType>(fXig);
    IndexVectorX erg = geometry::TetrahedralAabbHierarchy(problem.mesh.X, problem.mesh.E)
                           .NearestPrimitivesToPoints(CQ.Xg)
                           .first;
    rhog      = problem.rhoe(erg);
    mug       = problem.lame.row(0)(erg);
    lambdag   = problem.lame.row(1)(erg);
    auto cXig = fem::ReferencePositions(CM, CQ.eg, CQ.Xg);
    Ncg       = fem::ShapeFunctionsAt<VolumeMesh::ElementType>(cXig);
    GNcg      = fem::ShapeFunctionGradientsAt(CM, CQ.eg, cXig);
    xfg.resize(3, rhog.size());
}

void Restriction::Apply(Index iters, Level const& lf, Level& lc)
{
    // Compute target positions at quad.pts.
    VolumeMesh const& FM = lf.mesh;
    MatrixX const& xf    = lf.x;
    tbb::parallel_for(Index(0), efg.size(), [&](Index g) {
        auto e     = efg(g);
        auto inds  = FM.E(Eigen::placeholders::all, e);
        auto Nf    = Nfg.col(g);
        xfg.col(g) = xf(Eigen::placeholders::all, inds) * Nf;
    });
    // Minimize mass-weighted shape matching energy
    VolumeMesh const& CM    = lc.mesh;
    CageQuadrature const& Q = lc.Qcage;
    auto const [ptr, adj]   = std::tie(lc.ptr, lc.adj);
    MatrixX& xc             = lc.x;
    for (auto k = 0; k < iters; ++k)
    {
        auto nPartitions = ptr.size() - 1;
        for (Index p = 0; p < nPartitions; ++p)
        {
            auto pBegin = ptr(p);
            auto pEnd   = ptr(p + 1);
            tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                Index i     = adj(kp);
                auto gBegin = Q.GVGp(i);
                auto gEnd   = Q.GVGp(i + 1);
                using math::linalg::mini::SMatrix;
                using math::linalg::mini::SVector;
                using math::linalg::mini::Zeros;
                using math::linalg::mini::FromEigen;
                using math::linalg::mini::ToEigen;
                SMatrix<Scalar, 3, 3> Hi = Zeros<Scalar, 3, 3>();
                SVector<Scalar, 3> gi    = Zeros<Scalar, 3, 1>();
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index g             = Q.GVGg(kg);
                    Index ilocal        = Q.GVGilocal(kg);
                    Index e             = Q.eg(g);
                    Scalar wg           = Q.wg(g);
                    IndexVector<4> inds = CM.E.col(e);
                    SMatrix<Scalar, 3, 4> xce =
                        FromEigen(xc(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                    bool bSingular = Q.sg(g);
                    if (not bSingular)
                    {
                        Scalar rho            = rhog(g);
                        SVector<Scalar, 4> Nc = FromEigen(Ncg.col(g).head<4>());
                        SVector<Scalar, 3> x  = FromEigen(xfg.col(g).head<3>());
                        kernels::AccumulateShapeMatchingEnergy(ilocal, wg, rho, xce, Nc, x, gi, Hi);
                    }
                    else
                    {
                        SMatrix<Scalar, 4, 3> GNce = FromEigen(GNcg.block<4, 3>(0, 3 * g));
                        physics::StableNeoHookeanEnergy<3> Psi{};
                        Scalar mu     = mug(g);
                        Scalar lambda = lambdag(g);
                        kernels::AccumulateElasticEnergy(
                            ilocal,
                            wg,
                            Psi,
                            mu,
                            lambda,
                            xce,
                            GNce,
                            gi,
                            Hi);
                    }
                }
                // Commit descent step
                if (std::abs(Determinant(Hi)) < Scalar(1e-8))
                    return;
                SVector<Scalar, 3> dx = -(Inverse(Hi) * gi);
                xc.col(i) += ToEigen(dx);
            });
        }
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat