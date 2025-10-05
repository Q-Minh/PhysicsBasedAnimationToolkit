#include "ShapeFunctions.h"

#include "Jacobian.h"
#include "Mesh.h"
#include "Tetrahedron.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/polynomial/Basis.h"

#include <doctest/doctest.h>

TEST_CASE("[fem] ShapeFunctions")
{
    using namespace pbat;

    // Cube mesh
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
    common::ForRange<1, 4>([&]<auto QuadratureOrder>() {
        auto constexpr kOrder = 1;
        auto constexpr kDims  = 3;
        using Element         = fem::Tetrahedron<kOrder>;
        using Mesh            = fem::Mesh<Element, kDims>;

        Mesh const mesh{V, C};

        MatrixX const detJe             = fem::DeterminantOfJacobian<QuadratureOrder>(mesh);
        auto const numberOfElements     = mesh.E.cols();
        auto constexpr kNodesPerElement = Element::kNodes;
        MatrixX const gradNe            = fem::ShapeFunctionGradients<QuadratureOrder>(mesh);
        CHECK_EQ(gradNe.rows(), kNodesPerElement);
        CHECK_EQ(
            gradNe.cols(),
            Mesh::kDims * Element::QuadratureType<QuadratureOrder>::kPoints * numberOfElements);
    });
    common::ForRange<1, 4>([&]<auto PolynomialOrder>() {
        auto constexpr kDims = 3;
        using Element        = fem::Tetrahedron<PolynomialOrder>;
        using Mesh           = fem::Mesh<Element, kDims>;
        Mesh const mesh{V, C};

        // Arrange to evaluate shape functions at nodes.
        Matrix<Element::kDims, Element::kNodes> const XiRef =
            common::ToEigen(Element::Coordinates)
                .reshaped(Element::kDims, Element::kNodes)
                .template cast<Scalar>() /
            static_cast<Scalar>(Element::kOrder);

        auto const numberOfElements = mesh.E.cols();
        MatrixX Xi(Element::kDims, numberOfElements * Element::kNodes);
        for (auto e = 0; e < numberOfElements; ++e)
        {
            Xi.block<Element::kDims, Element::kNodes>(0, e * Element::kNodes) = XiRef;
        }

        // Compute shape functions at evaluation points
        MatrixX const N = fem::ShapeFunctionsAt<Element>(Xi);

        // Assert
        Scalar constexpr zero = 1e-15;
        CHECK_EQ(N.rows(), Element::kNodes);
        CHECK_EQ(N.cols(), Xi.cols());
        for (auto e = 0; e < numberOfElements; ++e)
        {
            Matrix<Element::kNodes, Element::kNodes> const Ne =
                N.block<Element::kNodes, Element::kNodes>(0, e * Element::kNodes);
            // Check that shape functions satisfy Kronecker delta property
            Matrix<Element::kNodes, Element::kNodes> const dij =
                Ne - Matrix<Element::kNodes, Element::kNodes>::Identity();
            Scalar const error = dij.squaredNorm();
            CHECK_LE(error, zero);
        }
    });
}

TEST_CASE("[fem] ShapeFunctionGradients")
{
    using namespace pbat;
    auto constexpr kOrder = 1;
    using ElementType     = fem::Tetrahedron<kOrder>;
    auto constexpr kDims  = ElementType::kDims;
    auto constexpr kNodes = ElementType::kNodes;
    // Some scaled and translated tetrahedron
    Matrix<kDims, 4> X;
    // clang-format off
    X << 0., 1., 0., 0.,
         0., 0., 1., 0.,
         0., 0., 0., 1.;
    // clang-format on
    Scalar constexpr scale          = 2.;
    Vector<kDims> const translation = Vector<kDims>::Ones();
    X *= scale;
    X.colwise() += translation;
    // We will test the gradients at barycenter
    Vector<kDims> const Xi{0.25, 0.25, 0.25};
    Vector<kDims + 1> BXi{};
    BXi(0)                = 1. - Xi.sum();
    BXi.segment(1, kDims) = Xi;

    Matrix<kNodes, kDims> const GP = fem::ElementShapeFunctionGradients<ElementType>(Xi, X);

    // Numerically compute basis functions and their gradients.
    // We know that the basis functions are interpolating polynomials,
    // thus we simply need to solve for the polynomials which yield the
    // Kronecker delta at nodes, i.e.:
    // P(X_i)^T a_j = \delta_{ij} ,
    // where a_j is the j^{th} column of some matrix of polynomial coefficients A.
    // This amounts to computing the inverse of P(X)^T .
    math::polynomial::MonomialBasis<kDims, kOrder> const P{};
    Matrix<kNodes, kNodes> PXT{};
    for (auto i = 0; i < kNodes; ++i)
        PXT.row(i) = P.eval(X.col(i)).transpose();
    Matrix<kNodes, kNodes> const A = PXT.inverse();
    // Knowing that the i^{th} basis function is a_i^T P(X),
    // its gradient is thus a_i^T \nabla P(X).
    Matrix<kNodes, kDims> const GPnumeric = (P.derivatives(X * BXi) * A).transpose();
    Matrix<kNodes, kDims> const GPerror   = GP - GPnumeric;
    Scalar const GPerrorMagnitude         = GPerror.squaredNorm();
    auto constexpr eps                    = 1e-14;
    CHECK_LE(GPerrorMagnitude, eps);
}

TEST_CASE("[fem] ShapeFunctionGradientsAt")
{
    using namespace pbat;

    // Cube mesh
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
    common::ForRange<1, 4>([&]<auto PolynomialOrder>() {
        auto constexpr kDims = 3;
        using ElementType    = fem::Tetrahedron<PolynomialOrder>;
        using MeshType       = fem::Mesh<ElementType, kDims>;
        MeshType const mesh{V, C};
        auto constexpr kQuadratureOrder = PolynomialOrder;
        using QuadratureRuleType = typename ElementType::template QuadratureType<kQuadratureOrder>;
        auto constexpr kQuadPts  = QuadratureRuleType::kPoints;
        auto const numberOfElements = mesh.E.cols();
        MatrixX Xi(kDims, kQuadPts * numberOfElements);
        IndexVectorX Ei(kQuadPts * numberOfElements);
        auto const Xg = common::ToEigen(QuadratureRuleType::points)
                            .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                            .template bottomRows<ElementType::kDims>();
        for (auto e = 0; e < numberOfElements; ++e)
        {
            Ei.segment<kQuadPts>(e * kQuadPts).setConstant(e);
            for (auto g = 0; g < kQuadPts; ++g)
            {
                Xi.col(e * kQuadPts + g) = Xg.col(g);
            }
        }
        MatrixX const GNe         = fem::ShapeFunctionGradientsAt(mesh, Ei, Xi);
        MatrixX const GNeExpected = fem::ShapeFunctionGradients<kQuadratureOrder>(mesh);
        Scalar const GNeError     = (GNe - GNeExpected).squaredNorm();
        Scalar constexpr zero     = 1e-15;
        CHECK_LE(GNeError, zero);
    });
}