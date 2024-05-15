#include "pba/fem/MassMatrix.h"

#include "pba/common/ConstexprFor.h"
#include "pba/fem/Mesh.h"
#include "pba/fem/Tetrahedron.h"
#include "pba/math/LinearOperator.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] MassMatrix")
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

    common::ForRange<1, 3>([&]<auto kOrder>() {
        common::ForRange<1, 4>([&]<auto OutDims> {
            using Element        = fem::Tetrahedron<kOrder>;
            auto constexpr kDims = 3;
            using Mesh           = fem::Mesh<Element, kDims>;
            Mesh mesh(V, C);
            auto const N          = mesh.X.cols();
            Scalar constexpr zero = 1e-10;
            auto const n          = N * OutDims;

            using MassMatrix = fem::MassMatrix<Mesh, OutDims>;
            CHECK(math::CLinearOperator<MassMatrix>);
            MassMatrix matrixFreeMass(mesh, 1.);

            SparseMatrix const M = matrixFreeMass.ToMatrix();
            CHECK_EQ(M.rows(), n);
            CHECK_EQ(M.cols(), n);

            // Check that matrix-free matrix multiplication has same result as matrix
            // multiplication
            VectorX const x = VectorX::Ones(n);
            VectorX yFree   = VectorX::Zero(n);
            matrixFreeMass.Apply(x, yFree);
            VectorX y           = M * x;
            Scalar const yError = (y - yFree).norm() / yFree.norm();
            CHECK_LE(yError, zero);

            // Check linearity M(kx) = kM(x)
            VectorX yInputScaled  = VectorX::Zero(n);
            VectorX yOutputScaled = VectorX::Zero(n);
            Scalar constexpr k    = -2.;
            matrixFreeMass.Apply(k * x, yInputScaled);
            matrixFreeMass.Apply(x, yOutputScaled);
            yOutputScaled *= k;
            Scalar const yLinearityError =
                (yInputScaled - yOutputScaled).norm() / yOutputScaled.norm();
            CHECK_LE(yLinearityError, zero);

            // TODO: We should probably check that the mass matrices actually have the
            // right values... But this is probably best done in a separate test.
        });
    });
}