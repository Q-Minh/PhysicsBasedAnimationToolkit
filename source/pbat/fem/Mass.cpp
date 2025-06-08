#include "Mass.h"

#include "Jacobian.h"
#include "Mesh.h"
#include "ShapeFunctions.h"
#include "Tetrahedron.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/LinearOperator.h"

#include <Eigen/Eigenvalues>
#include <doctest/doctest.h>

TEST_CASE("[fem] Mass")
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
        for (auto outDims = 1; outDims < 4; ++outDims)
        {
            using Element        = fem::Tetrahedron<kOrder>;
            auto constexpr kDims = 3;
            using Mesh           = fem::Mesh<Element, kDims>;
            Mesh mesh(V, C);
            auto const N          = mesh.X.cols();
            Scalar constexpr zero = 1e-10;
            auto const n          = N * outDims;
            auto const nElements  = mesh.E.cols();

            auto constexpr kQuadratureOrder = 2 * kOrder;
            auto const wg                   = fem::InnerProductWeights<kQuadratureOrder>(mesh);
            auto const nQuadPtsPerElement   = wg.rows();
            auto const Ng                   = fem::ShapeFunctions<Element, kQuadratureOrder>();
            auto const Neg                  = Ng.replicate(1, nElements);
            auto const rhog                 = VectorX::Constant(wg.size(), rho);
            auto const Meg = fem::ElementMassMatrices<Element>(Neg, wg.reshaped(), rhog);
            auto const eg  = IndexVectorX::LinSpaced(nElements, Index(0), nElements - 1)
                                .replicate(1, nQuadPtsPerElement)
                                .transpose()
                                .reshaped();
            auto const M = fem::MassMatrix<Eigen::ColMajor>(mesh, eg, Meg, outDims);
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
            fem::GemmMass(mesh, eg, Meg, outDims, x, yFree);
            VectorX y           = M * x;
            Scalar const yError = (y - yFree).norm() / yFree.norm();
            CHECK_LE(yError, zero);

            // Check linearity M(kx) = kM(x)
            VectorX yInputScaled  = VectorX::Zero(n);
            VectorX yOutputScaled = VectorX::Zero(n);
            Scalar constexpr k    = -2.;
            fem::GemmMass(mesh, eg, Meg, outDims, k * x, yInputScaled);
            fem::GemmMass(mesh, eg, Meg, outDims, x, yOutputScaled);
            yOutputScaled *= k;
            Scalar const yLinearityError =
                (yInputScaled - yOutputScaled).norm() / yOutputScaled.norm();
            CHECK_LE(yLinearityError, zero);

            // Check lumped mass
            VectorX lumpedMass = fem::LumpedMassMatrix(mesh, eg, Meg, outDims);
            CHECK_EQ(lumpedMass.size(), M.cols());
            for (auto i = 0; i < M.cols(); ++i)
            {
                Scalar const err = std::abs(lumpedMass(i) - M.col(i).sum());
                CHECK_LT(err, Scalar(1e-10));
            }

            // TODO: We should probably check that the mass matrices actually have the
            // right values... But this is probably best done in a separate test.
        }
    });
}