#include "Gradient.h"

#include "Jacobian.h"
#include "Laplacian.h"
#include "Mesh.h"
#include "ShapeFunctions.h"
#include "Tetrahedron.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/LinearOperator.h"

#include <doctest/doctest.h>
#include <unsupported/Eigen/KroneckerProduct>

TEST_CASE("[fem] Gradient")
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
        using Element                   = fem::Tetrahedron<kOrder>;
        auto constexpr kQuadratureOrder = kOrder > 1 ? kOrder - 1 : kOrder;
        auto constexpr kDims            = 3;
        Scalar constexpr zero           = 1e-10;
        using Mesh                      = fem::Mesh<Element, kDims>;
        Mesh mesh(V, C);

        auto const GNe             = fem::ShapeFunctionGradients<kQuadratureOrder>(mesh);
        auto const nQuadPtsPerElem = GNe.cols() / (kDims * mesh.E.cols());
        auto const eg = IndexVectorX::LinSpaced(mesh.E.cols(), Index(0), mesh.E.cols() - 1)
                            .replicate(1, nQuadPtsPerElem)
                            .transpose()
                            .reshaped();

        auto const GF = fem::MakeMatrixFreeGradient<Element, kDims>(mesh.E, mesh.X.cols(), eg, GNe);
        auto const G  = fem::GradientMatrix<Eigen::ColMajor>(GF);

        auto const n                = G.cols();
        auto const m                = G.rows();
        auto const numberOfElements = mesh.E.cols();
        auto const mExpected        = kDims * nQuadPtsPerElem * numberOfElements;
        auto const nExpected        = mesh.X.cols();
        CHECK_EQ(m, mExpected);
        CHECK_EQ(n, nExpected);

        VectorX const ones = VectorX::Ones(n);
        VectorX gradOnes   = VectorX::Zero(m);
        fem::GemmGradient(GF, ones, gradOnes);

        bool const bConstantFunctionHasZeroGradient = gradOnes.isZero(zero);
        CHECK(bConstantFunctionHasZeroGradient);
        VectorX const gradOnesMat                      = G * ones;
        bool const bConstantFunctionHasZeroGradientMat = gradOnesMat.isZero();
        CHECK(bConstantFunctionHasZeroGradientMat);

        // Compute Galerkin gradient
        CSRMatrix const N  = fem::ShapeFunctionMatrix<kQuadratureOrder>(mesh);
        CSCMatrix const NT = N.transpose();
        CSCMatrix Ik(kDims, kDims);
        Ik.setIdentity();
        CSCMatrix const NThat = Eigen::kroneckerProduct(NT, Ik);
        VectorX const Ihat    = fem::InnerProductWeights<kQuadratureOrder>(mesh)
                                 .reshaped()
                                 .template replicate<kDims, 1>();
        CSCMatrix const GG = NThat * Ihat.asDiagonal() * G;
        CHECK_EQ(GG.rows(), kDims * n);
        CHECK_EQ(GG.cols(), n);
        VectorX const galerkinGradOnes                      = GG * ones;
        bool const bConstantFunctionHasZeroGalerkinGradient = galerkinGradOnes.isZero(zero);
        CHECK(bConstantFunctionHasZeroGalerkinGradient);

        // Check that we can compute Laplacian via divergence of gradient
        CSCMatrix const GT       = G.transpose();
        CSCMatrix Lhat           = -GT * Ihat.asDiagonal() * G;
        VectorX const LhatValues = Lhat.coeffs();
        Lhat.coeffs().setOnes();
        VectorX const wg = fem::InnerProductWeights<kQuadratureOrder>(mesh).reshaped();
        CSCMatrix L      = fem::LaplacianMatrix<Element, kDims, Eigen::ColMajor>(
            mesh.E,
            mesh.X.cols(),
            eg,
            wg,
            GNe);
        VectorX const Lvalues = L.coeffs();
        L.coeffs().setOnes();
        Scalar const LsparsityError = (L - Lhat).squaredNorm();
        CHECK_LE(LsparsityError, zero);
        Lhat.coeffs()       = LhatValues;
        L.coeffs()          = Lvalues;
        Scalar const Lerror = (L - Lhat).squaredNorm();
        CHECK_LE(Lerror, zero);
    });
}