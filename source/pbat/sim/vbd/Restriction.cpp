#include "Restriction.h"

#include "Hierarchy.h"
#include "Kernels.h"
#include "Level.h"
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
    Level& Lc = H.levels[lcStl];
    DoApply(Lc, H.root.detHZero);
}

void Restriction::SetTargetShape(
    Eigen::Ref<IndexMatrixX const> const& Ef,
    Eigen::Ref<MatrixX const> const& xf)
{
    tbb::parallel_for(Index(0), xfg.cols(), [&](Index g) {
        Index ef   = efg(g);
        auto indsf = Ef.col(ef);
        auto xefg  = xf(Eigen::placeholders::all, indsf).block<3, 4>(0, 0);
        auto Nf    = Nfg.col(g).head<4>();
        xfg.col(g) = xefg * Nf;
    });
}

Scalar Restriction::DoApply(Level& Lc, Scalar detHZero) const
{
    auto nPartitions = Lc.C.ptr.size() - 1;
    VectorX energy(Lc.E.GVGg.size());
    for (auto iter = 0; iter < iterations; ++iter)
    {
        energy.setZero();
        for (auto p = 0; p < nPartitions; ++p)
        {
            auto pBegin = Lc.C.ptr[p];
            auto pEnd   = Lc.C.ptr[p + 1];
            for (auto kp = pBegin; kp < pEnd; ++kp)
            {
                // tbb::parallel_for(pBegin, pEnd, [&](Index kp) {
                using namespace math::linalg;
                using mini::FromEigen;
                using mini::SMatrix;
                using mini::SVector;
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
                        energy(kg) = Scalar(0.5) * wg * rhog * SquaredNorm(xcg * Ncg - xf);
                    }
                }
                if (std::abs(mini::Determinant(Hi)) < detHZero)
                {
                    continue;
                }
                SVector<Scalar, 3> dx = -(mini::Inverse(Hi) * gi);
                Lc.C.x.col(i) += ToEigen(dx);
                //});
            }
        }
    }
    return energy.sum();
}

} // namespace vbd
} // namespace sim
} // namespace pbat

#include <doctest/doctest.h>
#include <pbat/common/Eigen.h>
#include <pbat/fem/Jacobian.h>
#include <pbat/fem/Mesh.h>
#include <pbat/fem/ShapeFunctions.h>
#include <pbat/fem/Tetrahedron.h>
#include <pbat/geometry/TetrahedralAabbHierarchy.h>
#include <pbat/physics/HyperElasticity.h>
#include <span>

TEST_CASE("[sim][vbd] Restriction")
{
    using namespace pbat;
    // Cube mesh
    MatrixX VR(3, 8);
    IndexMatrixX CR(4, 5);
    // clang-format off
    VR << 0., 1., 0., 1., 0., 1., 0., 1.,
          0., 0., 1., 1., 0., 0., 1., 1.,
          0., 0., 0., 0., 1., 1., 1., 1.;
    CR << 0, 3, 5, 6, 0,
          1, 2, 4, 7, 5,
          3, 0, 6, 5, 3,
          5, 6, 0, 3, 6;
    // clang-format on
    // Center and create cage
    VR.colwise() -= VR.rowwise().mean();
    MatrixX VC      = Scalar(2) * VR;
    IndexMatrixX CC = CR;
    geometry::TetrahedralAabbHierarchy rbvh(VR, CR);
    geometry::TetrahedralAabbHierarchy cbvh(VC, CC);
    using sim::vbd::Level;
    using Energy = Level::Energy;
    using Cage   = Level::Cage;
    using Bus    = Level::RootParameterBus;
    // Construct quadratures
    using Element    = fem::Tetrahedron<1>;
    using Mesh       = fem::Mesh<Element, 3>;
    using BoolVector = Eigen::Vector<bool, Eigen::Dynamic>;
    Mesh MR(VR, CR);
    Mesh MC(VC, CC);
    VectorX wgR   = fem::InnerProductWeights<1>(MR).reshaped();
    MatrixX XgR   = MR.QuadraturePoints<1>();
    VectorX wgC   = fem::InnerProductWeights<3>(MC).reshaped();
    MatrixX XgC   = MC.QuadraturePoints<3>();
    VectorX rhogC = VectorX::Constant(wgC.size(), Scalar(1e3));
    Scalar Y = 1e6, nu = 0.45;
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    VectorX mug             = VectorX::Constant(wgC.size(), mu);
    VectorX lambdag         = VectorX::Constant(wgC.size(), lambda);
    // Construct Energy
    auto const graph = [](Index n, IndexMatrixX C, IndexMatrixX data) {
        using Triplet = Eigen::Triplet<Index, Index>;
        std::vector<Triplet> Gij{};
        for (auto c = 0; c < C.cols(); ++c)
            for (auto v = 0; v < C.rows(); ++v)
                Gij.push_back(Triplet{c, C(v, c), data(v, c)});

        Eigen::SparseMatrix<Index, Eigen::ColMajor, Index> G(C.cols(), n);
        G.setFromTriplets(Gij.begin(), Gij.end());
        return G;
    };
    Energy energy;
    {
        auto const [erg, sd] = rbvh.NearestPrimitivesToPoints(XgC);
        BoolVector sgC       = (sd.array() > Scalar(0)).eval();
        wgC                  = (sd.array() > Scalar(0)).select(Scalar(1e-14), wgC);
        energy.WithQuadrature(wgC, sgC);
        IndexVectorX ecg = cbvh.PrimitivesContainingPoints(XgC);
        MatrixX XigC     = fem::ReferencePositions(MC, ecg, XgC);
        MatrixX NcgC     = fem::ShapeFunctionsAt<Element>(XigC);
        energy.WithKineticEnergy(rhogC, NcgC);
        IndexMatrixX ervg(4, wgC.size());
        MatrixX Nrg(4, 4 * wgC.size());
        for (auto v = 0; v < 4; ++v)
        {
            MatrixX xv   = VR(Eigen::placeholders::all, CR(v, erg));
            ervg.row(v)  = cbvh.PrimitivesContainingPoints(xv);
            MatrixX Xivg = fem::ReferencePositions(MC, ervg.row(v), xv);
            MatrixX Nivg = fem::ShapeFunctionsAt<Element>(Xivg);
            Nrg(Eigen::placeholders::all, Eigen::seqN(v, wgC.size(), 4)) = Nivg;
        }
        MatrixX XigR = fem::ReferencePositions(MR, erg, XgC);
        MatrixX GNfg = fem::ShapeFunctionGradientsAt(MR, erg, XigR);
        MatrixX GNcg = fem::ShapeFunctionGradientsAt(MC, ecg, XigC);
        energy.WithPotentialEnergy(mug, lambdag, ervg, Nrg, GNfg, GNcg);
        IndexMatrixX CG = CC(Eigen::placeholders::all, ecg);
        IndexMatrixX ilocal =
            IndexVectorX::LinSpaced(Index(4), Index(0), Index(3)).replicate(1, CG.cols());
        IndexMatrixX elocal = ecg.transpose().replicate(4, 1);
        auto GVG            = graph(VC.cols(), CG, ilocal);
        std::span<Index> GVGp{GVG.outerIndexPtr(), static_cast<std::size_t>(GVG.outerSize() + 1)};
        std::span<Index> GVGg{GVG.innerIndexPtr(), static_cast<std::size_t>(GVG.nonZeros())};
        std::span<Index> GVGilocal{GVG.valuePtr(), static_cast<std::size_t>(GVG.nonZeros())};
        auto GVG2 = graph(VC.cols(), CG, elocal);
        std::span<Index> GVGe{GVG2.valuePtr(), static_cast<std::size_t>(GVG2.nonZeros())};
        energy.WithAdjacency(
            common::ToEigen(GVGp),
            common::ToEigen(GVGg),
            common::ToEigen(GVGe),
            common::ToEigen(GVGilocal));
    }
    Cage cage(
        VC,
        CC,
        IndexVectorX::LinSpaced(VC.cols() + 1, Index(0), VC.cols()),
        IndexVectorX::LinSpaced(VC.cols(), Index(0), VC.cols() - 1));
    auto const [erg, sd] = rbvh.NearestPrimitivesToPoints(XgC);
    MatrixX Xirg         = fem::ReferencePositions(MR, erg, XgC);
    Bus bus(erg, fem::ShapeFunctionsAt<Element>(Xirg));
    Level L(std::move(cage), std::move(energy.Construct()), std::move(bus));
    using sim::vbd::Restriction;
    Restriction R = Restriction()
                        .From(Index(-1))
                        .To(Index(0))
                        .WithFineShapeFunctions(L.RPB.erg, L.RPB.Nrg)
                        .Iterate(50)
                        .Construct();
    // Act
    MatrixX x = MR.X.colwise() + Vector<3>::Constant(Scalar(20.));
    R.SetTargetShape(MR.E, x);
    Scalar Eshape = R.DoApply(L, Scalar(1e-12));
    // Assert
    CHECK_FALSE(L.C.x.hasNaN());
    CHECK_LT(Eshape, Scalar(1e-5));
}
