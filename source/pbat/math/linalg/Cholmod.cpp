#include "Cholmod.h"
// clang-format off
#ifdef PBAT_USE_SUITESPARSE
#include <fmt/core.h>
#include <string>
// clang-format on

namespace pbat {
namespace math {
namespace linalg {

Cholmod::Cholmod() : mCholmodCommon{}, mCholmodL(NULL)
{
    cholmod_start(&mCholmodCommon);
}

MatrixX Cholmod::Solve(Eigen::Ref<MatrixX const> const& B) const
{
    if (mCholmodL->n != static_cast<size_t>(B.rows()))
    {
        std::string const what = fmt::format(
            "Expected right-hand side B to have {} rows, but got {} instead",
            mCholmodL->n,
            B.rows());
        throw std::invalid_argument(what);
    }

    cholmod_dense cholmod_B{};
    cholmod_B.nrow  = static_cast<size_t>(B.rows());
    cholmod_B.ncol  = static_cast<size_t>(B.cols());
    cholmod_B.nzmax = static_cast<size_t>(B.nonZeros());
    cholmod_B.d     = static_cast<size_t>(B.rows());
    cholmod_B.x     = const_cast<Scalar*>(B.data());
    cholmod_B.xtype = CHOLMOD_REAL;
    cholmod_B.dtype = CHOLMOD_DOUBLE;

    cholmod_dense* cholmod_X = cholmod_solve(
        CHOLMOD_A,
        mCholmodL,
        &cholmod_B,
        const_cast<cholmod_common*>(&mCholmodCommon));

    return Eigen::Map<MatrixX const>(
        static_cast<Scalar*>(cholmod_X->x),
        static_cast<Eigen::Index>(mCholmodL->n),
        B.cols());
}

Cholmod::~Cholmod()
{
    Deallocate();
    cholmod_finish(&mCholmodCommon);
}

void Cholmod::Deallocate()
{
    if (mCholmodL != NULL)
    {
        cholmod_free_factor(&mCholmodL, &mCholmodCommon);
    }
}

} // namespace linalg
} // namespace math
} // namespace pbat

    #include <doctest/doctest.h>

TEST_CASE("[math][linalg] Cholmod")
{
    using namespace pbat;
    // Arrange
    auto constexpr n    = 10;
    auto constexpr zero = 1e-15;
    auto constexpr m    = 3;
    MatrixX const X     = MatrixX::Random(n, m);
    MatrixX const R     = MatrixX::Random(n, n);
    CSCMatrix A         = (R.transpose() * R).sparseView();
    A                   = A.triangularView<Eigen::Lower>();
    MatrixX const B     = A.selfadjointView<Eigen::Lower>() * X;

    math::linalg::Cholmod LLT{};
    SUBCASE("Can solve SPD linear systems via Analyze+Factorize")
    {
        LLT.Analyze(A);
        bool const bFactorized = LLT.Factorize(A);
        CHECK(bFactorized);
        MatrixX const Xcomputed = LLT.Solve(B);
        Scalar const error      = (X - Xcomputed).squaredNorm();
        CHECK_LE(error, zero);
    }
    SUBCASE("Can solve SPD linear systems via Compute")
    {
        bool const bFactorized = LLT.Compute(A);
        CHECK(bFactorized);
        MatrixX const Xcomputed = LLT.Solve(B);
        Scalar const error      = (X - Xcomputed).squaredNorm();
        CHECK_LE(error, zero);
    }
    SUBCASE("Can update Cholesky factors")
    {
        CSCMatrix const U      = MatrixX::Random(n, m).sparseView();
        bool const bFactorized = LLT.Compute(A);
        CHECK(bFactorized);
        bool const bFactorizationUpdated = LLT.Update(U);
        CHECK(bFactorizationUpdated);
        CSCMatrix UT            = U.transpose();
        MatrixX const Bup       = (A.selfadjointView<Eigen::Lower>()) * X + U * UT * X;
        MatrixX const Xcomputed = LLT.Solve(Bup);
        Scalar const error      = (X - Xcomputed).squaredNorm();
        CHECK_LE(error, zero);
    }
    SUBCASE("Can downdate Cholesky factors")
    {
        CSCMatrix const U      = MatrixX::Zero(n, m).sparseView();
        bool const bFactorized = LLT.Compute(A);
        CHECK(bFactorized);
        bool const bFactorizationUpdated = LLT.Downdate(U);
        CHECK(bFactorizationUpdated);
        MatrixX const Xcomputed = LLT.Solve(B);
        Scalar const error      = (X - Xcomputed).squaredNorm();
        CHECK_LE(error, zero);
    }
}

#endif // PBAT_USE_SUITESPARSE