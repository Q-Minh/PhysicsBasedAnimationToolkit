#include "Quadrature.h"

#include "pbat/common/ArgSort.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/MassMatrix.h"
#include "pbat/fem/Triangle.h"
#include "pbat/geometry/MeshBoundary.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/math/MomentFitting.h"

#include <algorithm>
#include <numeric>
#include <tuple>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

template <class TDerivedEG, class TDerivedWG, class TDerivedXG>
std::tuple<IndexVectorX, VectorX, MatrixX> PatchQuadrature(
    VolumeMesh const& CM,
    Eigen::MatrixBase<TDerivedEG> const& eg1,
    Eigen::MatrixBase<TDerivedWG> const& wg1,
    Eigen::MatrixBase<TDerivedXG> const& Xg1,
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
    // Compute "negligible" 1-pt quadrature on coarse empty elements
    VectorX wg2 = fem::InnerProductWeights<1>(CM).reshaped()(eg2);
    wg2 *= zeroTetVolume;
    MatrixX Xg2 = CM.QuadraturePoints<1>()(Eigen::placeholders::all, eg2);
    // Combine embedded mesh's quadrature + the negligible quadrature
    MatrixX Xg(3, Xg1.cols() + Xg2.cols());
    VectorX wg(wg1.size() + wg2.size());
    IndexVectorX eg(eg1.size() + eg2.size());
    Xg << Xg1, Xg2;
    wg << wg1, wg2;
    eg << eg1, eg2;
    return std::make_tuple(eg, wg, Xg);
}

CageQuadrature::CageQuadrature(
    VolumeMesh const& FM,
    VolumeMesh const& CM,
    ECageQuadratureStrategy eStrategy)
    : Xg(), wg(), sg(), eg(), GVGp(), GVGg(), GVGilocal()
{
    geometry::TetrahedralAabbHierarchy fbvh(FM.X, FM.E);
    geometry::TetrahedralAabbHierarchy cbvh(CM.X, CM.E);

    switch (eStrategy)
    {
        case ECageQuadratureStrategy::CageMesh: {
            auto constexpr kCoarsePolynomialOrder = 3;
            // Simply use the symmetric simplex polynomial quadrature rule of the coarse mesh
            Xg = CM.QuadraturePoints<kCoarsePolynomialOrder>();
            eg = IndexVectorX::LinSpaced(CM.E.cols(), Index(0), CM.E.cols() - 1)
                     .transpose()
                     .replicate(Xg.cols() / CM.E.cols(), 1)
                     .reshaped();
            wg = fem::InnerProductWeights<kCoarsePolynomialOrder>(CM).reshaped();
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
                Xg1(Eigen::placeholders::all, eorder));
            break;
        }
        case ECageQuadratureStrategy::PolynomialSubCellIntegration: {
            // Make sure to have over-determined moment fitting systems, i.e. aim for
            // #quad.pts. >= 2|p|, where |p| is the size of the polynomial basis of order p.
            auto constexpr kPolynomialOrder                    = 1;
            auto constexpr kPolynomialOrderForSufficientPoints = 3;
            // Compute quadrature points via symmetric simplex quadrature rule on coarse mesh
            Xg = CM.QuadraturePoints<kPolynomialOrderForSufficientPoints>();
            eg = IndexVectorX::LinSpaced(CM.E.cols(), Index(0), CM.E.cols() - 1)
                     .transpose()
                     .replicate(Xg.cols() / CM.E.cols(), 1)
                     .reshaped();
            // Compute quadrature weights via moment fitting
            VectorX fwg               = fem::InnerProductWeights<1>(FM).reshaped();
            MatrixX fXg               = FM.QuadraturePoints<1>();
            IndexVectorX Sf           = cbvh.PrimitivesContainingPoints(fXg);
            MatrixX fXi               = fem::ReferencePositions(CM, Sf, fXg);
            MatrixX cXi               = fem::ReferencePositions(CM, eg, Xg);
            bool const bEvaluateError = true;
            auto const bMaxIterations = 10;
            VectorX error;
            auto const nSimplices = CM.E.cols();
            std::tie(wg, error)   = math::TransferQuadrature<kPolynomialOrder>(
                eg,
                cXi,
                Sf,
                fXi,
                fwg,
                nSimplices,
                bEvaluateError,
                bMaxIterations);
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
                Xg(Eigen::placeholders::all, validQuadPts));
            break;
        }
    }
    // Find singular quadrature points
    sg = (fbvh.PrimitivesContainingPoints(Xg).array() < 0);
    // Compute vertex-quad.pt. adjacency
    auto G = graph::MeshAdjacencyMatrix(CM.E(Eigen::placeholders::all, eg), CM.X.cols());
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
    using LinearTriangle = fem::Triangle<1>;
    using SurfaceMesh    = fem::Mesh<LinearTriangle, 3>;
    auto [V, F]          = geometry::SimplexMeshBoundary(FM.E, FM.X.cols());
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
    auto G = graph::MeshAdjacencyMatrix(CM.E(Eigen::placeholders::all, eg), CM.X.cols());
    G      = G.transpose();
    std::tie(GVGp, GVGg, GVGilocal) = graph::MatrixToAdjacency(G);
}

DirichletQuadrature::DirichletQuadrature(
    VolumeMesh const& FM,
    VolumeMesh const& CM,
    Eigen::Ref<VectorX const> const& m,
    Eigen::Ref<IndexVectorX const> const& dbcs)
    : Xg(), wg(), eg(), GVGp(), GVGg(), GVGilocal()
{
    geometry::TetrahedralAabbHierarchy cbvh(CM.X, CM.E);
    Xg = FM.X(Eigen::placeholders::all, dbcs);
    wg = m(dbcs);
    eg = cbvh.PrimitivesContainingPoints(Xg);
    // Compute vertex-quad.pt. adjacency
    auto G = graph::MeshAdjacencyMatrix(CM.E(Eigen::placeholders::all, eg), CM.X.cols());
    G      = G.transpose();
    std::tie(GVGp, GVGg, GVGilocal) = graph::MatrixToAdjacency(G);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][multigrid] Quadrature")
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
    MatrixX VC      = Scalar(1.1) * VR;
    IndexMatrixX CC = CR;

    // Act
    using sim::vbd::multigrid::CageQuadrature;
    using sim::vbd::multigrid::ECageQuadratureStrategy;
    using sim::vbd::multigrid::ESurfaceQuadratureStrategy;
    using sim::vbd::multigrid::SurfaceQuadrature;
    using VolumeMesh = sim::vbd::multigrid::VolumeMesh;
    VolumeMesh FM(VR, CR);
    VolumeMesh CM(VC, CC);
    CageQuadrature Qcage(FM, CM, ECageQuadratureStrategy::PolynomialSubCellIntegration);
    SurfaceQuadrature Qsurf(
        FM,
        CM,
        ESurfaceQuadratureStrategy::EmbeddedVertexSinglePointQuadrature);

    // Assert
    bool const bCageWeightsNonNegative = (Qcage.wg.array() >= Scalar(0)).all();
    bool const bSurfWeightsNonNegative = (Qsurf.wg.array() >= Scalar(0)).all();
    CHECK(bCageWeightsNonNegative);
    CHECK(bSurfWeightsNonNegative);
    Scalar const cageQuadEmbeddedVolumeError = std::abs(Qcage.wg.sum() - Scalar(1));
    CHECK_LT(cageQuadEmbeddedVolumeError, Scalar(1e-5));
    Scalar const surfQuadSurfaceAreaError = std::abs(Qsurf.wg.sum() - Scalar(6));
    CHECK_LT(surfQuadSurfaceAreaError, Scalar(1e-10));
}