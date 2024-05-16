#include "pba/math/LinearOperator.h"

#include <doctest/doctest.h>

namespace pba {
namespace test {

struct IdentityOperator
{
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y)
    {
        assert(x.rows() == n);
        assert(y.rows() == n);
        assert(x.cols() == y.cols());
        y += x;
    }

    pba::SparseMatrix ToMatrix() const
    {
        pba::SparseMatrix I(n, n);
        I.setIdentity();
        return I;
    }

    Index OutputDimensions() const { return n; }
    Index InputDimensions() const { return n; }

    int n;
};

} // namespace test
} // namespace pba

TEST_CASE("[math] LinearOperator")
{
    using namespace pba;
    CHECK(math::CLinearOperator<test::IdentityOperator>);

    using CompositeLinearOperator = math::CompositeLinearOperator<
        test::IdentityOperator,
        test::IdentityOperator,
        test::IdentityOperator>;

    auto constexpr nOperators = 3;
    auto constexpr n          = 5;
    auto constexpr zero       = 0.;
    test::IdentityOperator I{n};
    CompositeLinearOperator cop{I, test::IdentityOperator{n}, I};

    Vector<n> const x = Vector<n>::Ones();
    Vector<n> y       = Vector<n>::Zero();
    cop.Apply(x, y);

    Vector<n> const yExpected = Vector<n>::Constant(static_cast<Scalar>(nOperators));
    Scalar const yError       = (yExpected - y).norm() / yExpected.norm();
    CHECK_LE(yError, zero);

    SparseMatrix const I3    = cop.ToMatrix();
    Scalar const matrixError = (I3 - Matrix<n, n>::Identity().sparseView() * 3.).squaredNorm();
    CHECK_LE(matrixError, zero);

    Vector<n> yScaled = Vector<n>::Zero();
    Scalar const k    = -2.;
    cop.Apply(k * x, yScaled);
    Vector<n> const yScaledExpected = Vector<n>::Constant(static_cast<Scalar>(nOperators) * k);
    Scalar const yScaledError       = (yScaledExpected - yScaled).norm() / yScaledExpected.norm();
    CHECK_LE(yScaledError, zero);
}