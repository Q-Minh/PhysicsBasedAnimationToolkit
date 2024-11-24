#include "Restriction.h"

#include "Hierarchy.h"
#include "Kernels.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

void Restriction::Apply(Hierarchy& H)
{
    bool const bIsFineLevelRoot = lf < 0;
    auto lfStl                  = static_cast<std::size_t>(lf);
    auto lcStl                  = static_cast<std::size_t>(lc);
    IndexMatrixX const& Ef      = bIsFineLevelRoot ? H.root.T : H.levels[lfStl].C.E;
    MatrixX const& xf           = bIsFineLevelRoot ? H.root.x : H.levels[lfStl].C.x;
    // Precompute target (i.e. fine level) shapes
    xfg.resize(3, Nfg.cols());
    tbb::parallel_for(Index(0), xfg.cols(), [&](Index g) {
        Index ef   = efg(g);
        auto indsf = Ef.col(ef);
        auto xefg  = xf(Eigen::placeholders::all, indsf).block<3, 4>(0, 0);
        auto Nf    = Nfg.col(g).head<4>();
        xfg.col(g) = xefg * Nf;
    });
    // Fit coarse level to fine level, i.e. minimize shape matching energy
    Level& Lc        = H.levels[lcStl];
    auto nPartitions = Lc.P.ptr.size() - 1;
    for (auto iter = 0; iter < iterations; ++iter)
    {
        for (auto p = 0; p < nPartitions; ++p)
        {
            auto pBegin = Lc.P.ptr[p];
            auto pEnd   = Lc.P.ptr[p + 1];
            tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                using namespace math::linalg;
                using mini::SMatrix;
                using mini::SVector;
                using mini::FromEigen;
                using mini::ToEigen;

                Index i                  = Lc.P.adj[kp];
                SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                auto gBegin              = Lc.VG.GVGp[i];
                auto gEnd                = Lc.VG.GVGp[i + 1];
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index g        = Lc.VG.GVGg(kg);
                    Index e        = Lc.VG.GVGe(kg);
                    Index ilocal   = Lc.VG.GVGilocal(kg);
                    Scalar wg      = Lc.E.wg(g);
                    Scalar mug     = Lc.E.mug(g);
                    Scalar lambdag = Lc.E.lambdag(g);
                    auto inds      = Lc.C.E.col(e);
                    SMatrix<Scalar, 3, 4> xeg =
                        FromEigen(Lc.C.x(Eigen::placeholders::all, inds).block<3, 4>(0, 0));
                    bool const bSingular = Lc.E.sg(g);
                    // NOTE:
                    // Singular quadrature points are outside the domain, i.e. the target shape.
                    // There is no shape to match at these quad.pts. We simply ask that the part
                    // of the coarse cage outside the domain behave as an elastic model.
                    // Quadrature weights of these singular quad.pts. should be small enough that
                    // the elastic energy does not overpower the shape matching energy.
                    if (bSingular)
                    {
                        SMatrix<Scalar, 4, 3> GNeg = FromEigen(Lc.E.GNcg.block<4, 3>(0, g * 3));
                        SMatrix<Scalar, 3, 3> Feg  = xeg * GNeg;
                        physics::StableNeoHookeanEnergy<3> Psi{};
                        SVector<Scalar, 9> gF;
                        SMatrix<Scalar, 9, 9> HF;
                        Psi.gradAndHessian(Feg, mug, lambdag, gF, HF);
                        kernels::AccumulateElasticGradient(ilocal, wg, GNeg, gF, gi);
                        kernels::AccumulateElasticHessian(ilocal, wg, GNeg, HF, Hi);
                    }
                    else
                    {
                        Scalar rhog           = Lc.E.rhog(g);
                        SVector<Scalar, 4> Nc = FromEigen(Lc.E.Ncg.col(g).head<4>());
                        SVector<Scalar, 3> xf = FromEigen(xfg.col(g).head<3>());
                        auto xc               = xeg * Nc;
                        // Energy is 1/2 w_g rho_g || xc - xf ||_2^2
                        gi += (wg * rhog * Nc(ilocal)) * (xc - xf);
                        mini::Diag(Hi) += wg * rhog * Nc(ilocal) * Nc(ilocal);
                    }
                }
                SVector<Scalar, 3> dx = mini::Inverse(Hi) * gi;
                Lc.C.x.col(i) -= ToEigen(dx);
            });
        }
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat
