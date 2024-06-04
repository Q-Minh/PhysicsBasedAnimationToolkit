#include "pbat/fem/DivergenceVector.h"

#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/fem/Tetrahedron.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] DivergenceVector")
{
    using namespace pbat;
    // Cube tetrahedral mesh
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
    auto constexpr kOrder           = 1;
    auto constexpr kQuadratureOrder = kOrder;
    auto constexpr kDims            = 3;
    using ElementType               = fem::Tetrahedron<kOrder>;
    using MeshType                  = fem::Mesh<ElementType, kDims>;
    MeshType const M(V, C);
    auto const numberOfNodes = M.X.cols();
    MatrixX Fconstant(kDims, numberOfNodes);
    Fconstant.colwise() = Vector<3>{1., 2., 3.};

    MatrixX const GNe   = fem::ShapeFunctionGradients<kQuadratureOrder>(M);
    MatrixX const detJe = fem::DeterminantOfJacobian<kQuadratureOrder>(M);
    fem::DivergenceVector<MeshType, kDims, kQuadratureOrder> const divFconstant{
        M,
        detJe,
        GNe,
        Fconstant};
    VectorX const divFconstantVector = divFconstant.ToVector();
    CHECK_EQ(divFconstantVector.rows(), numberOfNodes);
    CHECK_EQ(divFconstantVector.cols(), 1);
}