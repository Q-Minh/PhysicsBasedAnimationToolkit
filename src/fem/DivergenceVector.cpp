#include "pba/fem/DivergenceVector.h"

#include "pba/common/ConstexprFor.h"
#include "pba/fem/Mesh.h"
#include "pba/fem/Tetrahedron.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] DivergenceVector")
{
    using namespace pba;
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
    auto constexpr kOrder = 1;
    auto constexpr kDims  = 3;
    using ElementType     = fem::Tetrahedron<kOrder>;
    using MeshType        = fem::Mesh<ElementType, kDims>;
    MeshType const M(V, C);
    auto const numberOfNodes = M.X.cols();
    MatrixX Fconstant(kDims, numberOfNodes);
    Fconstant.colwise() = Vector<3>{1., 2., 3.};
    fem::DivergenceVector<MeshType, kDims> const divFconstant{M, Fconstant};
    VectorX const divFconstantVector = divFconstant.ToVector();
    CHECK_EQ(divFconstantVector.rows(), numberOfNodes);
    CHECK_EQ(divFconstantVector.cols(), 1);
}