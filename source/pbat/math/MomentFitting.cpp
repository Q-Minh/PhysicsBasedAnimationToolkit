#include "MomentFitting.h"

#include "PolynomialBasis.h"
#include "SymmetricQuadratureRules.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/Eigen.h"

#include <doctest/doctest.h>

namespace pbat {
namespace math {
namespace test {

template <auto Dims, auto Order>
void TestFixedQuadrature(Scalar precision)
{
    pbat::math::OrthonormalPolynomialBasis<Dims, Order> P{};
    pbat::math::SymmetricSimplexPolynomialQuadratureRule<Dims, Order> Q{};
    auto Xg = pbat::common::ToEigen(Q.points).reshaped(Q.kDims + 1, Q.kPoints);
    auto wg = pbat::common::ToEigen(Q.weights);
    auto w =
        pbat::math::TransferQuadrature(P, Xg.bottomRows(Q.kDims), Xg.bottomRows(Q.kDims), wg, 20);
    CHECK_LT((w - wg).squaredNorm(), precision);
    CHECK((w.array() >= Scalar(0)).all());
};

} // namespace test
} // namespace math
} // namespace pbat

TEST_CASE("[math] MomentFitting")
{
    using namespace pbat;
    SUBCASE("Moment fitting reproduces fixed quadrature rule")
    {
        Scalar constexpr precision(1e-10);
        math::test::TestFixedQuadrature<1, 1>(precision);
        math::test::TestFixedQuadrature<1, 3>(precision);

        math::test::TestFixedQuadrature<2, 1>(precision);
        math::test::TestFixedQuadrature<2, 2>(precision);
        math::test::TestFixedQuadrature<2, 3>(precision);
        math::test::TestFixedQuadrature<2, 4>(precision);

        math::test::TestFixedQuadrature<3, 1>(precision);
        math::test::TestFixedQuadrature<3, 2>(precision);
        math::test::TestFixedQuadrature<3, 3>(precision);
        math::test::TestFixedQuadrature<3, 4>(precision);
    }
    SUBCASE("Moment fitting on mesh reproduces existing quadrature rule")
    {
        // Arrange
        auto constexpr kOrder         = 2;
        auto constexpr kDims          = 3;
        auto constexpr kMaxIterations = 20;
        math::SymmetricSimplexPolynomialQuadratureRule<kDims, kOrder> Q2{};
        auto X2Q = common::ToEigen(Q2.points).reshaped(kDims + 1, Q2.kPoints);
        auto w2Q = common::ToEigen(Q2.weights);

        IndexVectorX S1(8), S2(8);
        S2 << 0, 0, 0, 1, 1, 1, 1, 0;
        S1 = S2;
        MatrixX X1(kDims + 1, 8), X2(kDims + 1, 8);
        VectorX w2(8);
        X2 << X2Q.col(0), X2Q.col(1), X2Q.col(2), X2Q.col(0), X2Q.col(1), X2Q.col(2), X2Q.col(3),
            X2Q.col(3);
        w2 << w2Q(0), w2Q(1), w2Q(2), w2Q(0), w2Q(1), w2Q(2), w2Q(3), w2Q(3);
        X1 = X2;

        // Act
        bool constexpr bEvaluateError{true};
        auto [w1, error] = math::TransferQuadrature<kOrder>(
            S1,
            X1.bottomRows(kDims),
            S2,
            X2.bottomRows(kDims),
            w2,
            bEvaluateError,
            kMaxIterations);

        // Assert
        CHECK((w1.array() >= Scalar(0)).all());
        CHECK_LT(error.maxCoeff(), 1e-10);

        SUBCASE("Can also solve global sparse linear system for quadrature weights")
        {
            auto [M, B, P] = math::ReferenceMomentFittingSystems<kOrder>(
                S1,
                X1.bottomRows(kDims),
                S2,
                X2.bottomRows(kDims),
                w2);
            CSRMatrix GM = math::BlockDiagonalReferenceMomentFittingSystem(M, B, P);
            CHECK_EQ(GM.rows(), 2*math::OrthonormalPolynomialBasis<kDims, kOrder>::kSize);
            CHECK_EQ(GM.cols(), 8);
            CHECK_EQ(GM.nonZeros(), 2 * math::OrthonormalPolynomialBasis<kDims, kOrder>::kSize * 4);
        }
    }
}