#include "Restriction.h"

#include "Hierarchy.h"
#include "Kernels.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

Restriction& Restriction::From(Index lfIn)
{
    lf = lfIn;
    return *this;
}

Restriction& Restriction::To(Index lcIn)
{
    lc = lcIn;
    return *this;
}

Restriction& Restriction::WithFineShapeFunctions(
    Eigen::Ref<IndexVectorX const> const& efgIn,
    Eigen::Ref<MatrixX const> const& NfgIn)
{
    efg = efgIn;
    Nfg = NfgIn;
    xfg.resize(3, Nfg.cols());
    return *this;
}

Restriction& Restriction::Iterate(Index iterationsIn)
{
    iterations = iterationsIn;
    return *this;
}

Restriction& Restriction::Construct(bool bValidate)
{
    if (iterations < 1)
        iterations = 10;
        
    if (not bValidate)
        return *this;

    bool const bTransitionValid = (lc > lf);
    if (not bTransitionValid)
    {
        std::string const what = fmt::format("Expected lc > lf, but got lc={}, lf={}", lc, lf);
        throw std::invalid_argument(what);
    }
    bool const bShapeFunctionsAndElementsMatch = efg.size() == Nfg.cols();
    if (not bShapeFunctionsAndElementsMatch)
    {
        std::string const what = fmt::format(
            "Expected efg.size() == Nfg.cols(), but got dimensions efg={}, Nfg={}x{}",
            efg.size(),
            Nfg.rows(),
            Nfg.cols());
        throw std::invalid_argument(what);
    }

    return *this;
}

void Restriction::Apply(Hierarchy& H)
{
    bool const bIsFineLevelRoot = lf < 0;
    auto lfStl                  = static_cast<std::size_t>(lf);
    auto lcStl                  = static_cast<std::size_t>(lc);
    IndexMatrixX const& Ef      = bIsFineLevelRoot ? H.root.T : H.levels[lfStl].C.E;
    MatrixX const& xf           = bIsFineLevelRoot ? H.root.x : H.levels[lfStl].C.x;
    // Precompute target (i.e. fine level) shapes
    tbb::parallel_for(Index(0), xfg.cols(), [&](Index g) {
        Index ef   = efg(g);
        auto indsf = Ef.col(ef);
        auto xefg  = xf(Eigen::placeholders::all, indsf).block<3, 4>(0, 0);
        auto Nf    = Nfg.col(g).head<4>();
        xfg.col(g) = xefg * Nf;
    });
    // Fit coarse level to fine level, i.e. minimize shape matching energy
    Level& Lc        = H.levels[lcStl];
    auto nPartitions = Lc.C.ptr.size() - 1;
    for (auto iter = 0; iter < iterations; ++iter)
    {
        for (auto p = 0; p < nPartitions; ++p)
        {
            auto pBegin = Lc.C.ptr[p];
            auto pEnd   = Lc.C.ptr[p + 1];
            tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                using namespace math::linalg;
                using mini::SMatrix;
                using mini::SVector;
                using mini::FromEigen;
                using mini::ToEigen;

                Index i                  = Lc.C.adj[kp];
                SVector<Scalar, 3> gi    = mini::Zeros<Scalar, 3, 1>();
                SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
                auto gBegin              = Lc.E.GVGp[i];
                auto gEnd                = Lc.E.GVGp[i + 1];
                for (auto kg = gBegin; kg < gEnd; ++kg)
                {
                    Index g        = Lc.E.GVGg(kg);
                    Index e        = Lc.E.GVGe(kg);
                    Index ilocal   = Lc.E.GVGilocal(kg);
                    Scalar wg      = Lc.E.wg(g);
                    Scalar mug     = Lc.E.mug(g);
                    Scalar lambdag = Lc.E.lambdag(g);
                    auto inds      = Lc.C.E.col(e);
                    SMatrix<Scalar, 3, 4> xcg =
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
                        SMatrix<Scalar, 4, 3> GNcg = FromEigen(Lc.E.GNcg.block<4, 3>(0, g * 3));
                        physics::StableNeoHookeanEnergy<3> Psi{};
                        kernels::restriction::AccumulateSingularEnergy(
                            ilocal,
                            wg,
                            Psi,
                            mug,
                            lambdag,
                            xcg,
                            GNcg,
                            gi,
                            Hi);
                    }
                    else
                    {
                        Scalar rhog            = Lc.E.rhog(g);
                        SVector<Scalar, 4> Ncg = FromEigen(Lc.E.Ncg.col(g).head<4>());
                        SVector<Scalar, 3> xf  = FromEigen(xfg.col(g).head<3>());
                        kernels::restriction::AccumulateShapeMatchingEnergy(
                            ilocal,
                            wg,
                            rhog,
                            xcg,
                            Ncg,
                            xf,
                            gi,
                            Hi);
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
