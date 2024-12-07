#include "Quadrature.h"

#include "pbat/fem/Jacobian.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/fem/Triangle.h"
#include "pbat/geometry/MeshBoundary.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/math/MomentFitting.h"

#include <algorithm>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

CageQuadrature::CageQuadrature(
    Eigen::Ref<MatrixX const> const& X,
    Eigen::Ref<IndexMatrixX const> const& E,
    Eigen::Ref<MatrixX const> const& XC,
    Eigen::Ref<IndexMatrixX const> const& EC,
    ECageQuadratureStrategy eStrategy)
    : Xg(), wg(), sg(), eg(), GVGp(), GVGg(), GVGilocal()
{
    // Accelerate spatial queries
    geometry::TetrahedralAabbHierarchy cbvh(XC, EC);
    geometry::TetrahedralAabbHierarchy fbvh(X, E);
    // Compute linear FEM coarse mesh
    using LinearTetrahedron = fem::Tetrahedron<1>;
    using VolumeMesh        = fem::Mesh<LinearTetrahedron, 3>;
    VolumeMesh CM(XC, EC);
    if (eStrategy == ECageQuadratureStrategy::PolynomialSubCellIntegration)
    {
        // Make sure to have over-determined moment fitting systems, i.e. aim for
        // #quad.pts. >= 2p, where p is polynomial order.
        auto constexpr kPolynomialOrder                    = 1;
        auto constexpr kPolynomialOrderForSufficientPoints = 3;
        // Compute quadrature points via symmetric simplex quadrature rule on coarse mesh
        Xg = CM.QuadraturePoints<kPolynomialOrderForSufficientPoints>();
        sg = (fbvh.PrimitivesContainingPoints(Xg).array() < 0);
        eg = IndexVectorX::LinSpaced(EC.cols(), Index(0), EC.cols() - 1)
                 .transpose()
                 .replicate(Xg.cols() / EC.cols(), 1)
                 .reshaped();
        // Compute quadrature weights via moment fitting
        VolumeMesh FM(X, E);
        VectorX fwg               = fem::InnerProductWeights<1>(FM).reshaped();
        MatrixX fXg               = FM.QuadraturePoints<1>();
        IndexVectorX Sf           = cbvh.PrimitivesContainingPoints(fXg);
        MatrixX fXi               = fem::ReferencePositions(CM, Sf, fXg);
        MatrixX cXi               = fem::ReferencePositions(CM, eg, Xg);
        bool const bEvaluateError = true;
        auto const bMaxIterations = 10;
        VectorX error;
        auto const nSimplices = EC.cols();
        std::tie(wg, error)   = math::TransferQuadrature<
            kPolynomialOrder>(eg, cXi, Sf, fXi, fwg, nSimplices, bEvaluateError, bMaxIterations);
    }
    // Compute vertex-quad.pt. adjacency
    auto G = graph::MeshAdjacencyMatrix(EC(Eigen::placeholders::all, eg), XC.cols());
    G      = G.transpose();
    std::tie(GVGp, GVGg, GVGilocal) = graph::MatrixToAdjacency(G);
}

SurfaceQuadrature::SurfaceQuadrature(
    Eigen::Ref<MatrixX const> const& X,
    Eigen::Ref<IndexMatrixX const> const& E,
    Eigen::Ref<MatrixX const> const& XC,
    Eigen::Ref<IndexMatrixX const> const& EC,
    ESurfaceQuadratureStrategy eStrategy)
    : Xg(), wg(), eg(), GVGp(), GVGg(), GVGilocal()
{
    // Accelerate spatial queries
    geometry::TetrahedralAabbHierarchy cbvh(XC, EC);
    // Extract domain boundary
    using LinearTriangle = fem::Triangle<1>;
    using SurfaceMesh    = fem::Mesh<LinearTriangle, 3>;
    auto [V, F]          = geometry::SimplexMeshBoundary(E, X.cols());
    SurfaceMesh S(X, F);
    // Distribute 1-pt boundary triangle quadratures onto boundary vertices
    if (eStrategy == ESurfaceQuadratureStrategy::EmbeddedVertexSinglePointQuadrature)
    {
        VectorX FA = fem::InnerProductWeights<1>(S).reshaped();
        Xg         = X(Eigen::placeholders::all, V);
        wg.setZero(Xg.cols());
        for (auto d = 0; d < 3; ++d)
            wg(F.row(d)) += FA / Scalar(3);
        eg = cbvh.PrimitivesContainingPoints(Xg);
    }
    // Compute vertex-quad.pt. adjacency
    auto G = graph::MeshAdjacencyMatrix(EC(Eigen::placeholders::all, eg), XC.cols());
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
    CageQuadrature Qcage(VR, CR, VC, CC, ECageQuadratureStrategy::PolynomialSubCellIntegration);
    SurfaceQuadrature
        Qsurf(VR, CR, VC, CC, ESurfaceQuadratureStrategy::EmbeddedVertexSinglePointQuadrature);

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