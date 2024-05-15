#include "pba/fem/Mesh.h"

#include "pba/fem/Hexahedron.h"
#include "pba/fem/Tetrahedron.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] Mesh")
{
    using namespace pba;

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
            auto const kOrder                 = 1;
            auto const kDims                  = 3;
            using Element                     = fem::Tetrahedron<kOrder>;
            using Mesh                        = fem::Mesh<Element, kDims>;
            auto const kExpectedNumberOfNodes = 8;
            IndexVectorX nodeOrdering(kExpectedNumberOfNodes);
            nodeOrdering << 0, 1, 3, 5, 2, 6, 4, 7;

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