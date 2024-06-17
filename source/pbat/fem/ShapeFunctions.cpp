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
    common::ForRange<1, 4>([&]<auto QuadratureOrder>() {
        auto constexpr kOrder = 1;
        auto constexpr kDims  = 3;
        using Element         = fem::Tetrahedron<kOrder>;
        using Mesh            = fem::Mesh<Element, kDims>;

        Mesh const mesh{V, C};

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
    common::ForRange<1, 4>([&]<auto PolynomialOrder>() {
        auto constexpr kDims = 3;
        using Element        = fem::Tetrahedron<PolynomialOrder>;
        using Mesh           = fem::Mesh<Element, kDims>;
        Mesh const mesh{V, C};

        // Arrange to evaluate shape functions at nodes.
        Matrix<Element::kDims, Element::kNodes> const XiRef =
            common::ToEigen(Element::Coordinates)
                .reshaped(Element::kDims, Element::kNodes)
                .cast<Scalar>() /
            static_cast<Scalar>(Element::kOrder);

        auto const numberOfElements = mesh.E.cols();
        MatrixX Xi(Element::kDims, numberOfElements * Element::kNodes);
        for (auto e = 0; e < numberOfElements; ++e)
        {
            Xi.block<Element::kDims, Element::kNodes>(0, e * Element::kNodes) = XiRef;
        }

        // Compute shape functions at evaluation points
        MatrixX const N = fem::ShapeFunctionsAt<Element>(Xi);

        // Assert
        Scalar constexpr zero = 1e-15;
        CHECK_EQ(N.rows(), Element::kNodes);
        CHECK_EQ(N.cols(), Xi.cols());
        for (auto e = 0; e < numberOfElements; ++e)
        {
            Matrix<Element::kNodes, Element::kNodes> const Ne =
                N.block<Element::kNodes, Element::kNodes>(0, e * Element::kNodes);
            // Check that shape functions satisfy Kronecker delta property
            Matrix<Element::kNodes, Element::kNodes> const dij =
                Ne - Matrix<Element::kNodes, Element::kNodes>::Identity();
            Scalar const error = dij.squaredNorm();
            CHECK_LE(error, zero);
        }
    });
}
