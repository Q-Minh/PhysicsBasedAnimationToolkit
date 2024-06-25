#include "LaplacianMatrix.h"

#include "Jacobian.h"
#include "Mesh.h"
#include "ShapeFunctions.h"
#include "Tetrahedron.h"

#include <Eigen/Eigenvalues>
#include <doctest/doctest.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/LinearOperator.h>

TEST_CASE("[fem] LaplacianMatrix")
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

    common::ForRange<1, 3>([&]<auto kOrder>() {
        for (auto outDims = 1; outDims < 4; ++outDims)
        {
            using Element        = fem::Tetrahedron<kOrder>;
            auto constexpr kDims = Element::kDims;
            using Mesh           = fem::Mesh<Element, kDims>;
            Mesh mesh(V, C);
            auto const N                    = mesh.X.cols();
            Scalar constexpr zero           = 1e-10;
            auto const n                    = N * outDims;
            auto constexpr kQuadratureOrder = [&]() {
                if constexpr (kOrder == 1)
                    return 1;
                else
                    return (kOrder - 1) + (kOrder - 1);
            }();

            using LaplacianMatrix = fem::SymmetricLaplacianMatrix<Mesh, kQuadratureOrder>;
            MatrixX const detJe   = fem::DeterminantOfJacobian<kQuadratureOrder>(mesh);
            MatrixX const GNe     = fem::ShapeFunctionGradients<kQuadratureOrder>(mesh);
            CHECK(math::CLinearOperator<LaplacianMatrix>);
            LaplacianMatrix matrixFreeLaplacian(mesh, detJe, GNe, outDims);

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
            VectorX const x = VectorX::Random(n);
            VectorX yFree   = VectorX::Zero(n);
            matrixFreeLaplacian.Apply(x, yFree);
            VectorX y           = L * x;
            Scalar const yError = (y - yFree).squaredNorm();
            CHECK_LE(yError, zero);

            // Check linearity M(kx) = kM(x)
            VectorX yInputScaled  = VectorX::Zero(n);
            VectorX yOutputScaled = VectorX::Zero(n);
            Scalar constexpr k    = -2.;
            matrixFreeLaplacian.Apply(k * x, yInputScaled);
            matrixFreeLaplacian.Apply(x, yOutputScaled);
            yOutputScaled *= k;
            Scalar const yLinearityError =
                (yInputScaled - yOutputScaled).squaredNorm() / yOutputScaled.squaredNorm();
            CHECK_LE(yLinearityError, zero);

            // Laplacian of constant function should be 0
            VectorX const xconst = VectorX::Ones(n);
            VectorX yconst       = VectorX::Zero(n);
            matrixFreeLaplacian.Apply(xconst, yconst);
            CHECK_LE(yconst.squaredNorm(), zero);
        }
    });
}