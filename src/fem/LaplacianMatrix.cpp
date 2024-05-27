#include "pba/fem/LaplacianMatrix.h"

#include "pba/common/ConstexprFor.h"
#include "pba/fem/Mesh.h"
#include "pba/fem/Tetrahedron.h"
#include "pba/math/LinearOperator.h"

#include <Eigen/Eigenvalues>
#include <doctest/doctest.h>

TEST_CASE("[fem] LaplacianMatrix")
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
        using Element        = fem::Tetrahedron<kOrder>;
        auto constexpr kDims = Element::kDims;
        using Mesh           = fem::Mesh<Element, kDims>;
        Mesh mesh(V, C);
        auto const N          = mesh.X.cols();
        Scalar constexpr zero = 1e-10;
        auto const n          = N;

        using LaplacianMatrix = fem::SymmetricLaplacianMatrix<Mesh>;
        CHECK(math::CLinearOperator<LaplacianMatrix>);
        LaplacianMatrix matrixFreeLaplacian(mesh);

        CSCMatrix const L = matrixFreeLaplacian.ToMatrix();
        CHECK_EQ(L.rows(), n);
        CHECK_EQ(L.cols(), n);

        CSCMatrix const LT     = L.transpose();
        Scalar const Esymmetry = (LT - L).squaredNorm() / L.squaredNorm();
        CHECK_LE(Esymmetry, zero);

        Eigen::SelfAdjointEigenSolver<CSCMatrix> eigs(L);
        Scalar const maxEigenValue = eigs.eigenvalues().maxCoeff();
        bool const bIsNegativeSemiDefinite =
            (eigs.info() == Eigen::ComputationInfo::Success) and (maxEigenValue < zero);
        CHECK(bIsNegativeSemiDefinite);

        // Check that matrix-free matrix multiplication has same result as matrix
        // multiplication
        VectorX const x = VectorX::Ones(n);
        VectorX yFree   = VectorX::Zero(n);
        matrixFreeLaplacian.Apply(x, yFree);
        VectorX y           = L * x;
        Scalar const yError = (y - yFree).squaredNorm();
        CHECK_LE(yError, zero);
        // Laplacian of constant function should be zero
        CHECK_LE(yFree.squaredNorm(), zero);
        CHECK_LE(y.squaredNorm(), zero);

        // Check linearity M(kx) = kM(x)
        VectorX yInputScaled  = VectorX::Zero(n);
        VectorX yOutputScaled = VectorX::Zero(n);
        Scalar constexpr k    = -2.;
        matrixFreeLaplacian.Apply(k * x, yInputScaled);
        matrixFreeLaplacian.Apply(x, yOutputScaled);
        yOutputScaled *= k;
        Scalar const yLinearityError = (yInputScaled - yOutputScaled).norm() / yOutputScaled.norm();
        CHECK_LE(yLinearityError, zero);
    });
}