#include "Gradient.h"

#include "Jacobian.h"
#include "Mesh.h"
#include "ShapeFunctions.h"
#include "Tetrahedron.h"

#include <doctest/doctest.h>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/LinearOperator.h>

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

        MatrixX const detJe = fem::DeterminantOfJacobian<kQuadratureOrder>(mesh);
        MatrixX const GNe   = fem::ShapeFunctionGradients<kQuadratureOrder>(mesh);
        using GradientType  = fem::Gradient<Mesh, kQuadratureOrder>;
        CHECK(math::CLinearOperator<GradientType>);
        GradientType G{mesh, GNe};

        auto const n                = G.InputDimensions();
        auto const m                = G.OutputDimensions();
        auto const numberOfElements = mesh.E.cols();
        auto const kQuadPts         = detJe.rows();
        CHECK_EQ(m, kDims * kQuadPts * numberOfElements);

        VectorX const ones = VectorX::Ones(n);
        VectorX gradOnes   = VectorX::Zero(m);
        G.Apply(ones, gradOnes);

        bool const bConstantFunctionHasZeroGradient = gradOnes.isZero(zero);
        CHECK(bConstantFunctionHasZeroGradient);

        CSCMatrix const GM = G.ToMatrix();
        CHECK_EQ(GM.rows(), m);
        CHECK_EQ(GM.cols(), n);
        VectorX const gradOnesMat                      = GM * ones;
        bool const bConstantFunctionHasZeroGradientMat = gradOnesMat.isZero();
        CHECK(bConstantFunctionHasZeroGradientMat);

        // Compute Galerkin gradient
        CSRMatrix const N  = fem::ShapeFunctionMatrix<kQuadratureOrder>(mesh);
        CSCMatrix const NT = N.transpose();
        CSCMatrix Ik(kDims, kDims);
        Ik.setIdentity();
        CSCMatrix const NThat = Eigen::kroneckerProduct(NT, Ik);
        auto const Ihat       = detJe.reshaped().replicate<kDims, 1>();
        CSCMatrix const GG    = NThat * Ihat.asDiagonal() * GM;
        CHECK_EQ(GG.rows(), kDims * n);
        CHECK_EQ(GG.cols(), n);
        VectorX const galerkinGradOnes                      = GG * ones;
        bool const bConstantFunctionHasZeroGalerkinGradient = galerkinGradOnes.isZero(zero);
        CHECK(bConstantFunctionHasZeroGalerkinGradient);
    });
}