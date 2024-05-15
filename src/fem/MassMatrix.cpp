#include "pba/fem/MassMatrix.h"

#include "pba/fem/Mesh.h"
#include "pba/fem/Tetrahedron.h"
#include "pba/math/LinearOperator.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] MassMatrix")
{
    using namespace pba;

    auto const kOrder = 1;
    auto const kDims  = 3;
    using Element     = fem::Tetrahedron<kOrder>;
    using Mesh        = fem::Mesh<Element, kDims>;
    using MassMatrix  = fem::MassMatrix<Mesh, 1>;

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

    Mesh mesh(V, C);
    MassMatrix matrixFreeMass(mesh, 1.);
    Vector<5> const x = Vector<5>::Ones();
    Vector<5> y       = Vector<5>::Zero();
    matrixFreeMass.Apply(x, y);
    SparseMatrix const M = matrixFreeMass.ToMatrix();

    CHECK(math::CLinearOperator<MassMatrix>);
}