#include "MeshBoundary.h"

#include "pbat/common/Hash.h"

#include <algorithm>
#include <exception>
#include <fmt/format.h>
#include <string>
#include <unordered_map>

namespace pbat {
namespace geometry {

IndexMatrixX SimplexMeshBoundary(IndexMatrixX const& C, Index n)
{
    if (n < 0)
        n = C.maxCoeff() + 1;

    auto nSimplexNodes  = C.rows();
    auto nSimplexFacets = nSimplexNodes;
    if (nSimplexFacets < 3 or nSimplexFacets > 4)
    {
        std::string const what = fmt::format(
            "SimplexMeshBoundary expected triangle (3x|#elems|) or tetrahedral (4x|#elems|) input "
            "mesh, but got {}x{}",
            C.rows(),
            C.cols());
        throw std::invalid_argument(what);
    }
    auto nFacetNodes = nSimplexNodes - 1;
    auto nSimplices  = C.cols();
    IndexMatrixX F(nFacetNodes, nSimplexFacets * nSimplices);
    for (Index c = 0; c < nSimplices; ++c)
    {
        // Tetrahedra
        if (nSimplexFacets == 4)
        {
            auto Fc = F.block<3, 4>(0, c * 4);
            Fc.col(0) << C(0, c), C(1, c), C(3, c);
            Fc.col(1) << C(1, c), C(2, c), C(3, c);
            Fc.col(2) << C(2, c), C(0, c), C(3, c);
            Fc.col(3) << C(0, c), C(2, c), C(1, c);
        }
        // Triangles
        if (nSimplexFacets == 3)
        {
            auto Fc = F.block<2, 3>(0, c * 3);
            Fc.col(0) << C(0, c), C(1, c);
            Fc.col(1) << C(1, c), C(2, c);
            Fc.col(2) << C(2, c), C(0, c);
        }
    }
    // Sort face indices to identify duplicates next
    IndexMatrixX FS = F;
    for (Index f = 0; f < FS.cols(); ++f)
    {
        std::sort(FS.col(f).begin(), FS.col(f).end());
    }
    // Count face occurrences and pick out boundary facets
    auto fExtractBoundary = [&](auto const& FU) {
        Index nFacets{0};
        for (auto f = 0; f < FS.cols(); ++f)
            nFacets += (FU.at(FS.col(f)) == 1);
        IndexMatrixX B(nFacetNodes, nFacets);
        for (auto f = 0, b = 0; f < F.cols(); ++f)
            if (FU.at(FS.col(f)) == 1)
                B.col(b++) = F.col(f);
        return B;
    };
    IndexMatrixX B{};
    if (nSimplexFacets == 4)
    {
        std::unordered_map<IndexVector<3>, Index> FU{};
        FU.reserve(static_cast<std::size_t>(FS.cols()));
        for (Index f = 0; f < FS.cols(); ++f)
            ++FU[FS.col(f)];
        B = fExtractBoundary(FU);
    }
    if (nSimplexFacets == 3)
    {
        std::unordered_map<IndexVector<2>, Index> FU{};
        FU.reserve(static_cast<std::size_t>(FS.cols()));
        for (Index f = 0; f < FS.cols(); ++f)
            ++FU[FS.col(f)];
        B = fExtractBoundary(FU);
    }
    return B;
}

} // namespace geometry
} // namespace pbat

#include <doctest/doctest.h>
#include <pbat/fem/Jacobian.h>
#include <pbat/fem/Line.h>
#include <pbat/fem/Mesh.h>
#include <pbat/fem/Triangle.h>

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
    IndexMatrixX BC = geometry::SimplexMeshBoundary(C, 8);
    IndexMatrixX BF = geometry::SimplexMeshBoundary(F, 4);
    // Assert
    CHECK_EQ(BC.rows(), 3);
    CHECK_EQ(BC.cols(), 12);
    CHECK_EQ(BF.rows(), 2);
    CHECK_EQ(BF.cols(), 4);

    // Check facet areas/lengths
    using LinearTriangle = fem::Triangle<1>;
    using TriangleMesh   = fem::Mesh<LinearTriangle, 3>;
    TriangleMesh FM(V, BC);
    VectorX FA = fem::InnerProductWeights<1>(FM).reshaped();
    bool const bTrianglesHaveCorrectAreas =
        (FA.array() - Scalar(0.5)).square().sum() < Scalar(1e-10);
    CHECK(bTrianglesHaveCorrectAreas);
    using LinearLine = fem::Line<1>;
    using LineMesh   = fem::Mesh<LinearLine, 3>;
    LineMesh EM(V, BF);
    VectorX EL = fem::InnerProductWeights<1>(EM).reshaped();
    bool const bLineSegmentsHaveCorrectLengths =
        (EL.array() - Scalar(1)).square().sum() < Scalar(1e-10);
    CHECK(bLineSegmentsHaveCorrectLengths);
}