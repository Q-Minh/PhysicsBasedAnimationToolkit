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
    auto M  = pbat::math::ReferenceMomentFittingMatrix(P, Q);
    auto b  = pbat::Vector<decltype(P)::kSize>::Zero().eval();
    for (auto g = 0; g < Q.kPoints; ++g)
    {
        b += wg(g) * P.eval(Xg.col(g).segment<Dims>(1));
    }
    auto w = pbat::math::MomentFittedWeights(M, b, 20);
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
}