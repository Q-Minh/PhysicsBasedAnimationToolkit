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

    CHECK(math::CLinearOperator<MassMatrix>);
}