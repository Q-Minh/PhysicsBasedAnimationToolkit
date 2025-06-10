#include "Jacobian.h"

#include "Mesh.h"
#include "Tetrahedron.h"
#include "pbat/common/ConstexprFor.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] Jacobian")
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
    SUBCASE("Jacobian determinants")
    {
        common::ForRange<1, 4>([&]<auto PolynomialOrder>() {
            auto constexpr kDims = 3;
            using Element        = fem::Tetrahedron<PolynomialOrder>;
            using Mesh           = fem::Mesh<Element, kDims>;
            Mesh const mesh{V, C};
            common::ForRange<1, 4>([&]<auto QuadratureOrder>() {
                MatrixX const detJe = fem::DeterminantOfJacobian<QuadratureOrder>(mesh);
                CHECK_EQ(detJe.rows(), Element::template QuadratureType<QuadratureOrder>::kPoints);
                auto const numberOfElements = mesh.E.cols();
                CHECK_EQ(detJe.cols(), numberOfElements);
                bool const bIsStrictlyPositive = (detJe.array() > 0.).all();
                CHECK(bIsStrictlyPositive);
            });
        });
    }
    SUBCASE("Computing map from domain to reference space")
    {
        common::ForRange<1, 4>([&]<auto PolynomialOrder>() {
            auto constexpr kDims = 3;
            using Element        = fem::Tetrahedron<PolynomialOrder>;
            using Mesh           = fem::Mesh<Element, kDims>;
            Mesh const mesh{V, C};

            // Setup expected reference positions
            Matrix<Element::kDims, Element::kNodes> const XiExpected =
                common::ToEigen(Element::Coordinates)
                    .reshaped(Element::kDims, Element::kNodes)
                    .template cast<Scalar>() /
                static_cast<Scalar>(Element::kOrder);

            // Compute reference positions
            auto constexpr maxIterations = 2;
            Scalar constexpr eps         = 1e-10;
            auto eg = IndexVectorX::LinSpaced(mesh.E.cols(), 0, mesh.E.cols() - 1)
                          .replicate(1, Element::kNodes)
                          .transpose()
                          .reshaped()
                          .eval();
            MatrixX Xg               = mesh.X(Eigen::placeholders::all, mesh.E.reshaped());
            MatrixX const XiComputed = fem::ReferencePositions(mesh, eg, Xg, maxIterations, eps);

            // Assert
            CHECK_EQ(XiComputed.rows(), Element::kDims);
            CHECK_EQ(XiComputed.cols(), eg.size());
            for (auto e = 0; e < mesh.E.cols(); ++e)
            {
                auto const nodes = mesh.E.col(e);
                Matrix<Element::kDims, Element::kNodes> const XiComputedBlock =
                    XiComputed.block<Element::kDims, Element::kNodes>(0, e * Element::kNodes);
                Matrix<Element::kDims, Element::kNodes> const error = XiComputedBlock - XiExpected;
                Scalar const errorMagnitude                         = error.squaredNorm();
                CHECK_LE(errorMagnitude, eps);
            }
        });
    }
}
