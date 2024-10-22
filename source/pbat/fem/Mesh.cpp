#include "Mesh.h"

#include "Hexahedron.h"
#include "Tetrahedron.h"

#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>

TEST_CASE("[fem] Mesh")
{
    using namespace pbat;

    SUBCASE("Tetrahedral")
    {
        SUBCASE("Linear")
        {
            // Cube mesh
            MatrixX V(3, 8);
            IndexMatrixX C(4, 5);
            // clang-format off
            V << 0., 1., 0., 1., 0., 1., 0., 1.,
                 0., 0., 1., 1., 0., 0., 1., 1.,
                 0., 0., 0., 0., 1., 1., 1., 1.;
            C << 0, 3, 5, 6, 0,
                 1, 2, 4, 7, 5,
                 3, 0, 6, 5, 3,
                 5, 6, 0, 3, 6;
            // clang-format on
            auto constexpr kOrder             = 1;
            auto constexpr kDims              = 3;
            using Element                     = fem::Tetrahedron<kOrder>;
            using Mesh                        = fem::Mesh<Element, kDims>;
            auto const kExpectedNumberOfNodes = 8;
            IndexVectorX nodeOrdering(kExpectedNumberOfNodes);
            nodeOrdering << 0, 1, 2, 3, 4, 5, 6, 7;

            Mesh M(V, C);

            CHECK(fem::CElement<Element>);
            CHECK(fem::CMesh<Mesh>);
            CHECK_EQ(M.E.cols(), C.cols());
            CHECK_EQ(M.X.cols(), kExpectedNumberOfNodes);
            CHECK(V(Eigen::all, nodeOrdering) == M.X);
        }
        SUBCASE("Quadratic")
        {
            // 2 face-adjacent tet mesh
            MatrixX V(3, 5);
            IndexMatrixX C(4, 2);
            // clang-format off
            V << 0., 1., 0., 0., -1.,
                 0., 0., 1., 0., 0.,
                 0., 0., 0., 1., 0.;
            C << 0, 4,
                 1, 0,
                 2, 2,
                 3, 3;
            // clang-format on
            auto const kOrder                 = 2;
            auto const kDims                  = 3;
            using Element                     = fem::Tetrahedron<kOrder>;
            using Mesh                        = fem::Mesh<Element, kDims>;
            auto const kExpectedNumberOfNodes = 14;
            Mesh M(V, C);

            CHECK(fem::CElement<Element>);
            CHECK(fem::CMesh<Mesh>);
            CHECK_EQ(M.E.cols(), C.cols());
            CHECK_EQ(M.X.cols(), kExpectedNumberOfNodes);
        }
    }
    SUBCASE("Hexahedral")
    {
        // 2-Cube mesh
        MatrixX V(3, 12);
        IndexMatrixX C(8, 2);
        // clang-format off
        V << 0., 1., 0., 1., 0., 1., 0., 1., 2., 2., 2., 2.,
             0., 0., 1., 1., 0., 0., 1., 1., 0., 1., 0., 1.,
             0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1.;
        C << 0, 1,
             1, 8,
             2, 3,
             3, 9,
             4, 5,
             5, 10,
             6, 7,
             7, 11;
        // clang-format on
        SUBCASE("Linear")
        {
            auto const kOrder                     = 1;
            auto const kDims                      = 3;
            using Element                         = fem::Hexahedron<kOrder>;
            using Mesh                            = fem::Mesh<Element, kDims>;
            auto constexpr kExpectedNumberOfNodes = 12;
            IndexVectorX nodeOrdering(kExpectedNumberOfNodes);
            nodeOrdering << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11;
            Mesh M(V, C);

            CHECK(fem::CElement<Element>);
            CHECK(fem::CMesh<Mesh>);
            CHECK_EQ(M.E.cols(), C.cols());
            CHECK_EQ(M.X.cols(), kExpectedNumberOfNodes);
            CHECK(V(Eigen::all, nodeOrdering) == M.X);
        }
        SUBCASE("Quadratic")
        {
            auto const kOrder                     = 2;
            auto const kDims                      = 3;
            using Element                         = fem::Hexahedron<kOrder>;
            using Mesh                            = fem::Mesh<Element, kDims>;
            auto constexpr kExpectedNumberOfNodes = 45;
            Mesh M(V, C);

            CHECK(fem::CElement<Element>);
            CHECK(fem::CMesh<Mesh>);
            CHECK_EQ(M.E.cols(), C.cols());
            CHECK_EQ(M.X.cols(), kExpectedNumberOfNodes);
        }
    }
}

TEST_CASE("[fem] Mesh quadrature points")
{
    using namespace pbat;

    // Reference tetrahedron mesh
    Matrix<3, 4> V = Matrix<3, 4>::Zero();
    V.rightCols(3).setIdentity();
    IndexMatrix<4, 1> E;
    E.col(0).setLinSpaced(0, 3);

    // Translate tetrahedron
    Vector<3> const t{1., 2., 3.};
    V.colwise() += t;

    common::ForRange<1, 4>([&]<auto kPolynomialOrder>() {
        common::ForRange<1, 4>([&]<auto kQuadratureOrder>() {
            auto constexpr kDims = 3;
            using ElementType    = fem::Tetrahedron<kPolynomialOrder>;
            using MeshType       = fem::Mesh<ElementType, kDims>;

            MeshType mesh{V, E};
            MatrixX const Xg = mesh.template QuadraturePoints<kQuadratureOrder>();
            using QuadratureRuleType =
                typename ElementType::template QuadratureType<kQuadratureOrder>;
            auto constexpr kQuadPts = QuadratureRuleType::kPoints;
            auto const XgRef        = common::ToEigen(QuadratureRuleType::points)
                                   .reshaped(QuadratureRuleType::kDims + 1, kQuadPts)
                                   .template bottomRows<kDims>();
            CHECK_EQ(Xg.cols(), kQuadPts);
            CHECK_EQ(Xg.rows(), kDims);
            Scalar constexpr zero = 1e-10;
            for (auto g = 0; g < kQuadPts; ++g)
            {
                Vector<3> const XgComputed = Xg.col(g);
                Vector<3> const XgExpected = XgRef.col(g) + t;
                Scalar const XgError       = (XgComputed - XgExpected).squaredNorm();
                CHECK_LE(XgError, zero);
            }
        });
    });
}
