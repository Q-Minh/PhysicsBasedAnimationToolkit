#include "pbat/fem/DeformationGradient.h"

#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/Hexahedron.h"
#include "pbat/fem/Line.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/ShapeFunctionGradients.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/fem/Triangle.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] DeformationGradient")
{
    using namespace pbat;
    Scalar constexpr zero = 1e-15;

    SUBCASE("Tetrahedron")
    {
        // Translated unit tetrahedron
        Matrix<3, 4> V;
        IndexMatrix<4, 1> C;
        // clang-format off
        V << 0., 1., 0., 0.,
             0., 0., 1., 0.,
             0., 0., 0., 1.;
        C << 0, 1, 2, 3;
        // clang-format on
        V.colwise() += Vector<3>::Ones();

        common::ForRange<1, 3>([&]<auto kOrder>() {
            using ElementType    = fem::Tetrahedron<kOrder>;
            auto constexpr kDims = 3;
            using MeshType       = fem::Mesh<ElementType, kDims>;
            Vector<3> const Xi{0.25, 0.25, 0.25};
            // Adding translation should keep deformation gradient the same
            MeshType const M(V, C);
            Matrix<kDims, ElementType::kNodes> const x =
                M.X.colwise() + Vector<kDims>::Constant(3.);
            Matrix<ElementType::kNodes, ElementType::kDims> const GP =
                fem::ShapeFunctionGradients<ElementType>(Xi, V);
            Matrix<kDims, kDims> const F         = x * GP;
            Matrix<kDims, kDims> const Fexpected = Matrix<kDims, kDims>::Identity();
            Scalar const FError                  = (F - Fexpected).norm() / Fexpected.norm();
            CHECK_LE(FError, zero);
        });
    }
    SUBCASE("Triangle")
    {
        // Translated unit tetrahedron
        Matrix<3, 3> V;
        IndexMatrix<3, 1> C;
        // clang-format off
        V << 0., 1., 0.,
             0., 0., 1.,
             0., 0., 0.;
        C << 0, 1, 2;
        // clang-format on
        V.colwise() += Vector<3>::Ones();

        common::ForRange<1, 3>([&]<auto kOrder>() {
            using ElementType = fem::Triangle<kOrder>;
            // auto constexpr kInDims  = 2;
            auto constexpr kOutDims = 3;
            using MeshType          = fem::Mesh<ElementType, kOutDims>;
            Vector<2> const Xi{0.25, 0.25};
            // Adding translation should keep deformation gradient the same
            MeshType const M(V, C);
            Matrix<kOutDims, ElementType::kNodes> const x =
                M.X.colwise() + Vector<kOutDims>::Constant(3.);
            Matrix<ElementType::kNodes, kOutDims> const GP =
                fem::ShapeFunctionGradients<ElementType>(Xi, V);
            Matrix<kOutDims, kOutDims> const F = x * GP;

            Matrix<kOutDims, kOutDims> Fexpected;
            // NOTE: Why is this the expected deformation gradient for 2D triangle embedded in 3D?
            // It is translated in all 3 dimensions, but the deformation map still ignores the 3rd
            // dimension. Interesting.
            // clang-format off
            Fexpected << 1., 0., 0., 
                         0., 1., 0., 
                         0., 0., 0.;
            // clang-format on
            Scalar const FError = (F - Fexpected).norm() / Fexpected.norm();
            CHECK_LE(FError, zero);
        });
    }
}