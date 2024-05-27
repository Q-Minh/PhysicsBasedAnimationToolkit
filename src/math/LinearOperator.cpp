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

    pba::CSCMatrix ToMatrix() const
    {
        pba::CSCMatrix I(n, n);
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

    auto constexpr nOperators = 3;
    auto constexpr n          = 5;
    auto constexpr zero       = 0.;
    test::IdentityOperator I{n};
    auto const cop = math::ComposeLinearOperators(I, I, I);

    Vector<n> const x = Vector<n>::Ones();
    Vector<n> y       = Vector<n>::Zero();
    cop.Apply(x, y);

    Vector<n> const yExpected = Vector<n>::Constant(static_cast<Scalar>(nOperators));
    Scalar const yError       = (yExpected - y).norm() / yExpected.norm();
    CHECK_LE(yError, zero);

    // Our LinearOperator is supposed to be Eigen compatible. However, we only support Eigen matrix
    // operations on dynamic-size vectors/matrices. This is because internally, Eigen uses a
    // compile-time trick to select
    Vector<n> const xEigenExpected{x};
    Vector<n> const yEigen   = cop * xEigenExpected;
    Scalar const yEigenError = (yExpected - yEigen).norm() / yExpected.norm();
    CHECK_LE(yEigenError, zero);
    using CGSolver = Eigen::ConjugateGradient<
        decltype(cop),
        Eigen::Lower | Eigen::Upper,
        Eigen::IdentityPreconditioner>;
    CGSolver cg;
    cg.compute(cop);
    Vector<n> const xEigen   = cg.solve(yEigen);
    Scalar const xEigenError = (xEigenExpected - xEigen).norm() / xEigenExpected.norm();
    CHECK_LE(xEigenError, zero);

    CSCMatrix const I3         = cop.ToMatrix();
    CSCMatrix const I3expected = 3. * Matrix<n, n>::Identity().sparseView();
    Scalar const matrixError   = (I3 - I3expected).squaredNorm();
    CHECK_LE(matrixError, zero);

    Vector<n> yScaled = Vector<n>::Zero();
    Scalar const k    = -2.;
    cop.Apply(k * x, yScaled);
    Vector<n> const yScaledExpected = Vector<n>::Constant(static_cast<Scalar>(nOperators) * k);
    Scalar const yScaledError       = (yScaledExpected - yScaled).norm() / yScaledExpected.norm();
    CHECK_LE(yScaledError, zero);
}