#include "Quadrature.h"

#include "pbat/common/ArgSort.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/fem/Triangle.h"
#include "pbat/geometry/MeshBoundary.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/math/MomentFitting.h"

#include <algorithm>
#include <exception>
#include <fmt/format.h>
#include <numeric>
#include <tuple>

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

template <class TDerivedEG, class TDerivedWG, class TDerivedXG>
std::tuple<IndexVectorX, VectorX, MatrixX> PatchQuadrature(
    VolumeMesh const& CM,
    Eigen::MatrixBase<TDerivedEG> const& eg1,
    Eigen::MatrixBase<TDerivedWG> const& wg1,
    Eigen::MatrixBase<TDerivedXG> const& Xg1,
    int patchOrder       = 2,
    Scalar zeroTetVolume = Scalar(1e-6))
{
    // Find empty coarse elements
    IndexVectorX egcpy = eg1;
    {
        std::sort(egcpy.begin(), egcpy.end());
        auto it = std::unique(egcpy.begin(), egcpy.end());
        egcpy.conservativeResize(std::distance(egcpy.begin(), it));
    }
    IndexVectorX eg2{};
    {
        IndexVectorX eall(CM.E.cols());
        std::iota(eall.begin(), eall.end(), Index(0));
        eg2.resizeLike(eall);
        auto it =
            std::set_difference(eall.begin(), eall.end(), egcpy.begin(), egcpy.end(), eg2.begin());
        eg2.conservativeResize(std::distance(eg2.begin(), it));
    }
    // Return input quadrature if it was already fine
    bool const bNeedsPatching = eg2.size() > 0;
    if (not bNeedsPatching)
    {
        return std::make_tuple<IndexVectorX, VectorX, MatrixX>(eg1, wg1, Xg1);
    }
    // Compute "negligible" 1-pt quadrature on coarse empty elements
    VectorX wg2{};
    MatrixX Xg2{};
    common::ForRange<1, 7>([&]<auto kOrder>() {
        if (kOrder == patchOrder)
        {
            wg2 = fem::InnerProductWeights<kOrder>(CM)(Eigen::placeholders::all, eg2).reshaped();
            wg2 *= zeroTetVolume;
            auto nQuadPtsPerElem = wg2.size() / eg2.size();
            auto eg2q            = eg2.replicate(nQuadPtsPerElem, 1).reshaped();
            Xg2                  = CM.QuadraturePoints<kOrder>()(Eigen::placeholders::all, eg2q);
        }
    });
    // Combine embedded mesh's quadrature + the negligible quadrature
    MatrixX Xg(3, Xg1.cols() + Xg2.cols());
    VectorX wg(wg1.size() + wg2.size());
    IndexVectorX eg(eg1.size() + eg2.size());
    Xg << Xg1, Xg2;
    wg << wg1, wg2;
    eg << eg1, eg2;
    return std::make_tuple(eg, wg, Xg);
}

CageQuadratureParameters&
CageQuadratureParameters::WithStrategy(ECageQuadratureStrategy eStrategyIn)
{
    eStrategy = eStrategyIn;
    return *this;
}

CageQuadratureParameters& CageQuadratureParameters::WithCageMeshPointsOfOrder(int order)
{
    if (order < 1 or order > 6)
    {
        throw std::invalid_argument(
            fmt::format("Expected 1 <= order <= 6, but got order={}", order));
    }
    mCageMeshPointsOfOrder = order;
    return *this;
}

CageQuadratureParameters& CageQuadratureParameters::WithPatchCellPointsOfOrder(int order)
{
    if (order < 1 or order > 6)
    {
        throw std::invalid_argument(
            fmt::format("Expected 1 <= order <= 6, but got order={}", order));
    }
    mPatchCellPointsOfOrder = order;
    return *this;
}

CageQuadratureParameters& CageQuadratureParameters::WithPatchError(Scalar err)
{
    mPatchTetVolumeError = err;
    return *this;
}

CageQuadrature::CageQuadrature(
    VolumeMesh const& FM,
    VolumeMesh const& CM,
    CageQuadratureParameters const& params)
    : Xg(), wg(), sg(), eg(), Ncg(), GNcg(), efg(), Nfg(), GNfg(), GVGp(), GVGg(), GVGilocal()
{
    geometry::TetrahedralAabbHierarchy fbvh(FM.X, FM.E);
    geometry::TetrahedralAabbHierarchy cbvh(CM.X, CM.E);

    switch (params.eStrategy)
    {
        case ECageQuadratureStrategy::CageMesh: {
            // Simply use the symmetric simplex polynomial quadrature rule of the coarse mesh
            common::ForRange<1, 7>([&]<auto kCoarsePolynomialOrder>() {
                if (params.mCageMeshPointsOfOrder == kCoarsePolynomialOrder)
                {
                    Xg = CM.QuadraturePoints<kCoarsePolynomialOrder>();
                    wg = fem::InnerProductWeights<kCoarsePolynomialOrder>(CM).reshaped();
                    eg = IndexVectorX::LinSpaced(CM.E.cols(), Index(0), CM.E.cols() - 1)
                             .transpose()
                             .replicate(Xg.cols() / CM.E.cols(), 1)
                             .reshaped();
                }
            });
            break;
        }
        case ECageQuadratureStrategy::EmbeddedMesh: {
            MatrixX Xg1      = FM.QuadraturePoints<1>();
            IndexVectorX eg1 = cbvh.PrimitivesContainingPoints(Xg1);
            VectorX wg1      = fem::InnerProductWeights<1>(FM).reshaped();
            // Group quadrature points of same elements together to improve cache locality.
            IndexVectorX eorder =
                common::ArgSort(eg1.size(), [&](auto i, auto j) { return eg1(i) < eg1(j); });
            // Patch coarse elements that don't have any embedded quadrature point
            std::tie(eg, wg, Xg) = PatchQuadrature(
                CM,
                eg1(eorder),
                wg1(eorder),
                Xg1(Eigen::placeholders::all, eorder),
                params.mPatchCellPointsOfOrder,
                params.mPatchTetVolumeError);
            break;
        }
        case ECageQuadratureStrategy::PolynomialSubCellIntegration: {
            // Make sure to have over-determined moment fitting systems, i.e. aim for
            // #quad.pts. >= 2|p|, where |p| is the size of the polynomial basis of order p.
            auto constexpr kPolynomialOrder = 1;
            common::ForRange<1, 7>([&]<auto kPolynomialOrderForSufficientPoints>() {
                // Compute quadrature points via symmetric simplex quadrature rule on coarse
                // mesh
                if (params.mCageMeshPointsOfOrder == kPolynomialOrderForSufficientPoints)
                    Xg = CM.QuadraturePoints<kPolynomialOrderForSufficientPoints>();
            });
            auto nQuadPtsPerElem = Xg.cols() / CM.E.cols();
            // Elements containing quad.pts. are ordered thus
            eg = IndexVectorX::LinSpaced(CM.E.cols(), Index(0), CM.E.cols() - 1)
                     .transpose()
                     .replicate(nQuadPtsPerElem, 1)
                     .reshaped();
            // Compute quadrature weights via moment fitting
            MatrixX fXg                = FM.QuadraturePoints<1>();
            VectorX fwg                = fem::InnerProductWeights<1>(FM).reshaped();
            IndexVectorX Sf            = cbvh.PrimitivesContainingPoints(fXg);
            MatrixX fXi                = fem::ReferencePositions(CM, Sf, fXg, 10);
            MatrixX cXi                = fem::ReferencePositions(CM, eg, Xg, 10);
            auto const nSimplices      = CM.E.cols();
            bool const bEvaluateError  = true;
            Index const bMaxIterations = 10 * nQuadPtsPerElem;
            Scalar const precision     = 1e-10 * fwg.maxCoeff();
            VectorX error;
            std::tie(wg, error) = math::TransferQuadrature<kPolynomialOrder>(
                eg,
                cXi,
                Sf,
                fXi,
                fwg,
                nSimplices,
                bEvaluateError,
                bMaxIterations,
                precision);
            // Find all non-negligible quadrature points (i.e. quadrature weight > 0)
            Index const nQuadPts = wg.size();
            std::vector<Index> validQuadPts{};
            validQuadPts.reserve(static_cast<std::size_t>(nQuadPts));
            for (Index g = 0; g < nQuadPts; ++g)
                if (wg(g) > Scalar(0))
                    validQuadPts.push_back(g);
            // Remove negligible quadrature points and patch the quadrature
            std::tie(eg, wg, Xg) = PatchQuadrature(
                CM,
                eg(validQuadPts),
                wg(validQuadPts),
                Xg(Eigen::placeholders::all, validQuadPts),
                params.mPatchCellPointsOfOrder,
                params.mPatchTetVolumeError);
            break;
        }
    }
    // Find singular quadrature points
    VectorX sd{};
    std::tie(efg, sd) = fbvh.NearestPrimitivesToPoints(Xg);
    sg                = (sd.array() > Scalar(0));
    // Precompute shape functions and their gradients at quad.pts.
    auto cXig = fem::ReferencePositions(CM, eg, Xg);
    Ncg       = fem::ShapeFunctionsAt(CM, eg, cXig, true);
    GNcg      = fem::ShapeFunctionGradientsAt(CM, eg, cXig, true);
    auto fXig = fem::ReferencePositions(FM, efg, Xg);
    Nfg       = fem::ShapeFunctionsAt(FM, efg, fXig, true);
    GNfg      = fem::ShapeFunctionGradientsAt(FM, efg, fXig, true);
    // Compute vertex-quad.pt. adjacency
    IndexMatrixX ilocal = IndexVector<4>{0, 1, 2, 3}.replicate(1, eg.size());
    auto G = graph::MeshAdjacencyMatrix(CM.E(Eigen::placeholders::all, eg), ilocal, CM.X.cols());
    G      = G.transpose();
    std::tie(GVGp, GVGg, GVGilocal) = graph::MatrixToAdjacency(G);
}

SurfaceQuadrature::SurfaceQuadrature(
    VolumeMesh const& FM,
    VolumeMesh const& CM,
    ESurfaceQuadratureStrategy eStrategy)
    : Xg(), wg(), eg(), GVGp(), GVGg(), GVGilocal()
{
    geometry::TetrahedralAabbHierarchy cbvh(CM.X, CM.E);
    // Extract domain boundary
    auto [V, F] = geometry::SimplexMeshBoundary(FM.E, FM.X.cols());
    SurfaceMesh S(FM.X, F);
    // Distribute 1-pt boundary triangle quadratures onto boundary vertices
    if (eStrategy == ESurfaceQuadratureStrategy::EmbeddedVertexSinglePointQuadrature)
    {
        VectorX FA = fem::InnerProductWeights<1>(S).reshaped();
        Xg         = FM.X(Eigen::placeholders::all, V);
        wg.setZero(Xg.cols());
        for (auto d = 0; d < 3; ++d)
            wg(F.row(d)) += FA / Scalar(3);
        eg = cbvh.PrimitivesContainingPoints(Xg);
    }
    // Compute vertex-quad.pt. adjacency
    IndexMatrixX ilocal = IndexVector<4>{0, 1, 2, 3}.replicate(1, eg.size());
    auto G = graph::MeshAdjacencyMatrix(CM.E(Eigen::placeholders::all, eg), ilocal, CM.X.cols());
    G      = G.transpose();
    std::tie(GVGp, GVGg, GVGilocal) = graph::MatrixToAdjacency(G);
}

DirichletQuadrature::DirichletQuadrature(
    VolumeMesh const& FM,
    VolumeMesh const& CM,
    Eigen::Ref<VectorX const> const& m,
    Eigen::Ref<IndexVectorX const> const& dbcs)
    : Xg(), wg(), eg(), Ncg(), GVGp(), GVGg(), GVGilocal()
{
    geometry::TetrahedralAabbHierarchy cbvh(CM.X, CM.E);
    Xg  = FM.X(Eigen::placeholders::all, dbcs);
    wg  = m(dbcs);
    eg  = cbvh.PrimitivesContainingPoints(Xg);
    Ncg = fem::ShapeFunctionsAt(CM, eg, Xg);
    // Compute vertex-quad.pt. adjacency
    IndexMatrixX ilocal = IndexVector<4>{0, 1, 2, 3}.replicate(1, eg.size());
    auto G = graph::MeshAdjacencyMatrix(CM.E(Eigen::placeholders::all, eg), ilocal, CM.X.cols());
    G      = G.transpose();
    std::tie(GVGp, GVGg, GVGilocal) = graph::MatrixToAdjacency(G);
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat

#ifdef PBAT_WITH_PRECOMPILED_LARGE_MODELS
    #include "pbat/geometry/model/Armadillo.h"
#endif // PBAT_WITH_PRECOMPILED_LARGE_MODELS
#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][lod] Quadrature")
{
    using namespace pbat;
    using sim::vbd::lod::CageQuadrature;
    using sim::vbd::lod::CageQuadratureParameters;
    using sim::vbd::lod::ECageQuadratureStrategy;
    using sim::vbd::lod::ESurfaceQuadratureStrategy;
    using sim::vbd::lod::SurfaceQuadrature;
    using VolumeMesh  = sim::vbd::VolumeMesh;
    using SurfaceMesh = sim::vbd::SurfaceMesh;

    auto const ActAndAssert =
        [](MatrixX const& VR, IndexMatrixX const& CR, MatrixX const& VC, IndexMatrixX const& CC) {
            // Act
            VolumeMesh FM(VR, CR);
            VolumeMesh CM(VC, CC);
            CageQuadrature Qcage(
                FM,
                CM,
                CageQuadratureParameters{}.WithStrategy(
                    ECageQuadratureStrategy::PolynomialSubCellIntegration));
            SurfaceQuadrature Qsurf(
                FM,
                CM,
                ESurfaceQuadratureStrategy::EmbeddedVertexSinglePointQuadrature);

            // Assert
            bool const bCageWeightsNonNegative = (Qcage.wg.array() >= Scalar(0)).all();
            bool const bSurfWeightsNonNegative = (Qsurf.wg.array() >= Scalar(0)).all();
            CHECK(bCageWeightsNonNegative);
            CHECK(bSurfWeightsNonNegative);
            Scalar const expectedVolume = fem::InnerProductWeights<1>(FM).sum();
            Scalar const cageQuadEmbeddedVolumeError =
                std::abs(Qcage.wg.sum() - expectedVolume) / expectedVolume;
            CHECK_LT(cageQuadEmbeddedVolumeError, Scalar(1e-2));

            auto [FVR, FFR] = geometry::SimplexMeshBoundary(CR, VR.cols());
            SurfaceMesh SM(VR, FFR);
            Scalar const expectedSurfaceArea = fem::InnerProductWeights<1>(SM).sum();
            Scalar const surfQuadSurfaceAreaError =
                std::abs(Qsurf.wg.sum() - expectedSurfaceArea) / expectedSurfaceArea;
            CHECK_LT(surfQuadSurfaceAreaError, Scalar(1e-10));
        };

    SUBCASE("Cube")
    {
        auto [VR, CR] = geometry::model::Cube();
        // Center and create cage
        VR.colwise() -= VR.rowwise().mean();
        MatrixX VC      = Scalar(1.1) * VR;
        IndexMatrixX CC = CR;
        ActAndAssert(VR, CR, VC, CC);
    }
#ifdef PBAT_WITH_PRECOMPILED_LARGE_MODELS
    SUBCASE("Armadillo")
    {
        auto [VR, CR] = geometry::model::Armadillo(geometry::model::EMesh::Tetrahedral, Index(0));
        auto [VC, CC] = geometry::model::Armadillo(geometry::model::EMesh::Tetrahedral, Index(1));
        ActAndAssert(VR, CR, VC, CC);
    }
#endif // PBAT_WITH_PRECOMPILED_LARGE_MODELS
}