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

Restriction::Restriction(CageQuadrature const& CQ) : xfg()
{
    xfg.resize(3, CQ.Xg.cols());
}

void Restriction::Apply(Index iters, Level const& lf, Level& lc)
{
    DoApply(iters, lf.x, lf.mesh.E, lc);
}

Scalar Restriction::DoApply(
    Index iters,
    Eigen::Ref<MatrixX const> const& xf,
    Eigen::Ref<IndexMatrixX const> const& Ef,
    Level& lc)
{
    CageQuadrature const& Q = lc.Qcage;
    // Compute target positions at quad.pts.
    tbb::parallel_for(Index(0), Q.efg.size(), [&](Index g) {
        auto e     = Q.efg(g);
        auto inds  = Ef(Eigen::placeholders::all, e);
        auto Nf    = Q.Nfg.col(g);
        xfg.col(g) = xf(Eigen::placeholders::all, inds) * Nf;
    });
    // Minimize mass-weighted shape matching energy
    VolumeMesh const& CM  = lc.mesh;
    auto const [ptr, adj] = std::tie(lc.ptr, lc.adj);
    MatrixX& xc           = lc.x;
    VectorX energy(lc.Qcage.GVGg.size());
    for (auto k = 0; k < iters; ++k)
    {
        energy.setZero();
        auto nPartitions = ptr.size() - 1;
        for (Index p = 0; p < nPartitions; ++p)
        {
            auto pBegin = ptr(p);
            auto pEnd   = ptr(p + 1);
            tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                Index i     = adj(kp);
                auto gBegin = Q.GVGp(i);
                auto gEnd   = Q.GVGp(i + 1);
                using math::linalg::mini::FromEigen;
                using math::linalg::mini::SMatrix;
                using math::linalg::mini::SVector;
                using math::linalg::mini::ToEigen;
                using math::linalg::mini::Zeros;
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
                        Scalar rho            = lc.Ekinetic.rhog(g);
                        SVector<Scalar, 4> Nc = FromEigen(Q.Ncg.col(g).head<4>());
                        SVector<Scalar, 3> x  = FromEigen(xfg.col(g).head<3>());
                        energy(kg)            = kernels::AccumulateShapeMatchingEnergy(
                            ilocal,
                            wg,
                            rho,
                            xce,
                            Nc,
                            x,
                            gi,
                            Hi);
                    }
                    else
                    {
                        SMatrix<Scalar, 4, 3> GNce = FromEigen(Q.GNcg.block<4, 3>(0, 3 * g));
                        physics::StableNeoHookeanEnergy<3> Psi{};
                        Scalar mu     = lc.Epotential.mug(g);
                        Scalar lambda = lc.Epotential.lambdag(g);
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
                //if (std::abs(Determinant(Hi)) < Scalar(1e-8))
                //    return;
                SVector<Scalar, 3> dx = -(Inverse(Hi) * gi);
                xc.col(i) += ToEigen(dx);
            });
        }
    }
    return energy.maxCoeff();
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/geometry/model/Cube.h"
#ifdef PBAT_WITH_PRECOMPILED_LARGE_MODELS
    #include "pbat/geometry/model/Armadillo.h"
#endif // PBAT_WITH_PRECOMPILED_LARGE_MODELS

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][multigrid] Restriction")
{
    using namespace pbat;
    using sim::vbd::Data;
    using sim::vbd::multigrid::CageQuadrature;
    using sim::vbd::multigrid::CageQuadratureParameters;
    using sim::vbd::multigrid::Level;
    using sim::vbd::multigrid::Restriction;
    using sim::vbd::multigrid::VolumeMesh;

    auto const fActAndAssert = [](Index iters,
                                  VolumeMesh const& FM,
                                  VolumeMesh const& CM,
                                  Scalar translate,
                                  Scalar scale) {
        Data data = Data().WithVolumeMesh(FM.X, FM.E).Construct();
        Level lf{FM};
        Level lc = Level{CM}
                       .WithCageQuadrature(data, CageQuadratureParameters{})
                       .WithElasticEnergy(data)
                       .WithMomentumEnergy(data);
        Restriction R(lc.Qcage);
        // Translate and scale fine mesh
        lf.x.colwise() -= translate * Vector<3>::Ones();
        lf.x.array() *= scale;
        // Restrict coarse mesh to fine mesh
        Scalar energy    = R.DoApply(iters, lf.x, lf.mesh.E, lc);
        Scalar bboxDiag2 = (lf.x.rowwise().maxCoeff() - lf.x.rowwise().minCoeff()).squaredNorm();
        Scalar wgMax     = lc.Qcage.wg.maxCoeff();
        // Energy is max_g (1/2 w_g rho_g || xc - xf ||_2^2)
        Scalar const kLargestExpectedError = Scalar(1e-6) * bboxDiag2 * wgMax * data.rhoe.mean();
        CHECK_LT(energy, kLargestExpectedError);
    };

    SUBCASE("Cube")
    {
        auto [VR, CR] = geometry::model::Cube();
        // Center and create cage
        VR.colwise() -= VR.rowwise().mean();
        MatrixX VC      = Scalar(1.1) * VR;
        IndexMatrixX CC = CR;
        VolumeMesh FM(VR, CR);
        VolumeMesh CM(VC, CC);
        Index constexpr iters = 10;
        Scalar constexpr scale{5};
        Scalar constexpr translate{5};
        fActAndAssert(iters, FM, CM, translate, scale);
    }
#ifdef PBAT_WITH_PRECOMPILED_LARGE_MODELS
    SUBCASE("Armadillo")
    {
        auto [VR, CR] = geometry::model::Armadillo(geometry::model::EMesh::Tetrahedral, Index(0));
        auto [VC, CC] = geometry::model::Armadillo(geometry::model::EMesh::Tetrahedral, Index(1));
        VolumeMesh FM(VR, CR);
        VolumeMesh CM(VC, CC);
        Index constexpr iters = 150;
        Scalar constexpr scale{2.};
        Scalar constexpr translate{2.};
        fActAndAssert(iters, FM, CM, translate, scale);
    }
#endif // PBAT_WITH_PRECOMPILED_LARGE_MODELS
}
