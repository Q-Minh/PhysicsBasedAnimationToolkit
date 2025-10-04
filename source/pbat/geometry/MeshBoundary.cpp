#include "MeshBoundary.h"

namespace pbat::geometry {
} // namespace pbat::geometry

#include "pbat/fem/Line.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/MeshQuadrature.h"
#include "pbat/fem/Triangle.h"

#include <doctest/doctest.h>

TEST_CASE("[geometry] MeshBoundary")
{
    using namespace pbat;
    // Cube mesh
    MatrixX V(3, 8);
    IndexMatrixX C(4, 5);
    IndexMatrixX F(3, 2);
    // clang-format off
    V << 0., 1., 0., 1., 0., 1., 0., 1.,
        0., 0., 1., 1., 0., 0., 1., 1.,
        0., 0., 0., 0., 1., 1., 1., 1.;
    C << 0, 3, 5, 6, 0,
        1, 2, 4, 7, 5,
        3, 0, 6, 5, 3,
        5, 6, 0, 3, 6;
    F << 0, 2,
        1, 1, 
        2, 3;
    // clang-format on

    // Act
    auto [VC, BC] = geometry::SimplexMeshBoundary<Index>(C, 8);
    auto [VF, BF] = geometry::SimplexMeshBoundary<Index>(F, 4);
    // Assert
    CHECK_EQ(VC.size(), 8);
    CHECK_EQ(VF.size(), 4);
    CHECK_EQ(BC.rows(), 3);
    CHECK_EQ(BC.cols(), 12);
    CHECK_EQ(BF.rows(), 2);
    CHECK_EQ(BF.cols(), 4);

    // Check facet areas/lengths
    using LinearTriangle = fem::Triangle<1>;
    using TriangleMesh   = fem::Mesh<LinearTriangle, 3>;
    TriangleMesh FM(V, BC);
    VectorX FA = fem::MeshQuadratureWeights<1>(FM).reshaped();
    bool const bTrianglesHaveCorrectAreas =
        (FA.array() - Scalar{0.5}).square().sum() < Scalar{1e-10};
    CHECK(bTrianglesHaveCorrectAreas);
    using LinearLine = fem::Line<1>;
    using LineMesh   = fem::Mesh<LinearLine, 3>;
    LineMesh EM(V, BF);
    VectorX EL = fem::MeshQuadratureWeights<1>(EM).reshaped();
    bool const bLineSegmentsHaveCorrectLengths =
        (EL.array() - Scalar{1}).square().sum() < Scalar{1e-10};
    CHECK(bLineSegmentsHaveCorrectLengths);
}