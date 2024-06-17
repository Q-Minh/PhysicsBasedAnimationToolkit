#include "Jacobian.h"

#include "Mesh.h"
#include "Tetrahedron.h"

#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>

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
