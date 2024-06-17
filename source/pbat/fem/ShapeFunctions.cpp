#include "ShapeFunctions.h"

#include "Jacobian.h"
#include "Mesh.h"
#include "Tetrahedron.h"

#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>

TEST_CASE("[fem] ShapeFunctions")
{
    using namespace pbat;

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
    auto constexpr kOrder = 1;
    auto constexpr kDims  = 3;
    using Element         = fem::Tetrahedron<kOrder>;
    using Mesh            = fem::Mesh<Element, kDims>;

    Mesh const mesh{V, C};
    common::ForRange<1, 4>([&]<auto QuadratureOrder>() {
        MatrixX const detJe         = fem::DeterminantOfJacobian<QuadratureOrder>(mesh);
        MatrixX const intNe         = fem::IntegratedShapeFunctions<QuadratureOrder>(mesh, detJe);
        auto const numberOfElements = mesh.E.cols();
        auto constexpr kNodesPerElement = Element::kNodes;
        CHECK_EQ(intNe.rows(), kNodesPerElement);
        CHECK_EQ(intNe.cols(), numberOfElements);
        bool const bIsStrictlyPositive = (intNe.array() > 0.).all();
        CHECK(bIsStrictlyPositive);
        MatrixX const gradNe = fem::ShapeFunctionGradients<QuadratureOrder>(mesh);
        CHECK_EQ(gradNe.rows(), kNodesPerElement);
        CHECK_EQ(
            gradNe.cols(),
            Mesh::kDims * Element::QuadratureType<QuadratureOrder>::kPoints * numberOfElements);
    });
}
