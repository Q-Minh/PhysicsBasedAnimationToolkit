#include "pbat/fem/Gradient.h"

#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/math/LinearOperator.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] GalerkinGradient")
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
        auto constexpr kQuadratureOrder = 2 * kOrder - 1;
        auto constexpr kDims            = 3;
        Scalar constexpr zero           = 1e-10;
        using Mesh                      = fem::Mesh<Element, kDims>;
        Mesh mesh(V, C);

        MatrixX const detJe = fem::DeterminantOfJacobian<kQuadratureOrder>(mesh);
        MatrixX const GNe   = fem::ShapeFunctionGradients<kQuadratureOrder>(mesh);
        using GradientType  = fem::GalerkinGradient<Mesh, kQuadratureOrder>;
        CHECK(math::CLinearOperator<GradientType>);
        GradientType G{mesh, detJe, GNe};

        auto const n = G.InputDimensions();
        auto const m = G.OutputDimensions();
        CHECK_EQ(m, kDims * n);

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
    });
}