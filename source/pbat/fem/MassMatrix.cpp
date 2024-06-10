#include "MassMatrix.h"

#include "Jacobian.h"
#include "Mesh.h"
#include "Tetrahedron.h"

#include <Eigen/Eigenvalues>
#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/LinearOperator.h>

TEST_CASE("[fem] MassMatrix")
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

    Scalar constexpr rho = 1.;
    common::ForRange<1, 3>([&]<auto kOrder>() {
        common::ForRange<1, 4>([&]<auto OutDims> {
            using Element        = fem::Tetrahedron<kOrder>;
            auto constexpr kDims = 3;
            using Mesh           = fem::Mesh<Element, kDims>;
            Mesh mesh(V, C);
            auto const N          = mesh.X.cols();
            Scalar constexpr zero = 1e-10;
            auto const n          = N * OutDims;

            auto constexpr kQuadratureOrder = 2 * kOrder;
            using MassMatrix                = fem::MassMatrix<Mesh, OutDims, kQuadratureOrder>;
            CHECK(math::CLinearOperator<MassMatrix>);
            MatrixX const detJe = fem::DeterminantOfJacobian<kQuadratureOrder>(mesh);
            MassMatrix matrixFreeMass(mesh, detJe, rho);

            CSCMatrix const M = matrixFreeMass.ToMatrix();
            CHECK_EQ(M.rows(), n);
            CHECK_EQ(M.cols(), n);

            CSCMatrix const MT     = M.transpose();
            Scalar const Esymmetry = (MT - M).squaredNorm() / M.squaredNorm();
            CHECK_LE(Esymmetry, zero);
            Eigen::SelfAdjointEigenSolver<CSCMatrix> eigs(M);
            bool const bIsPositiveDefinite = (eigs.info() == Eigen::ComputationInfo::Success) and
                                             (eigs.eigenvalues().minCoeff() >= 0.);
            CHECK(bIsPositiveDefinite);

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