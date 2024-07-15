#include "Cholmod.h"

#include <fmt/core.h>
#include <string>

namespace pbat {
namespace math {
namespace linalg {

Cholmod::Cholmod() : mCholmodCommon{}, mCholmodA(NULL), mCholmodL(NULL), mbFactorized{false}
{
    cholmod_start(&mCholmodCommon);
}

void Cholmod::analyze()
{
    if (mCholmodL != NULL)
        return;

    mCholmodL = cholmod_analyze(mCholmodA, &mCholmodCommon);

    if (mCholmodL == NULL)
        throw std::runtime_error("Symbolic analysis of Cholesky factor failed");

    mbFactorized = false;
}

bool Cholmod::factorize()
{
    if (mCholmodA == NULL)
        throw std::runtime_error("Cannot factorize matrix A, which was not set.");

    if (mCholmodL == NULL)
        analyze();

    int const ec = cholmod_factorize(mCholmodA, mCholmodL, &mCholmodCommon);
    if (ec == 1)
    {
        mbFactorized = true;
        return true;
    }
    return false;
}

MatrixX Cholmod::solve(Eigen::Ref<MatrixX const> const& B) const
{
    if (!mbFactorized)
    {
        throw std::runtime_error("Cannot solve unfactorized system.");
    }

    if (mCholmodA->ncol != static_cast<size_t>(B.rows()))
    {
        std::string const what = fmt::format(
            "Expected right-hand side B to have {} rows, but got {} instead",
            mCholmodA->ncol,
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
        static_cast<Eigen::Index>(mCholmodA->nrow),
        B.cols());
}

Cholmod::~Cholmod()
{
    deallocate();
    cholmod_finish(&mCholmodCommon);
}

void Cholmod::deallocate()
{
    if (mCholmodA != NULL)
    {
        cholmod_free_sparse(&mCholmodA, &mCholmodCommon);
    }
    if (mCholmodL != NULL)
    {
        cholmod_free_factor(&mCholmodL, &mCholmodCommon);
    }
    mbFactorized = false;
}

} // namespace linalg
} // namespace math
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[math][linalg] Cholmod")
{
    using namespace pbat;
    auto constexpr n = 10;
    MatrixX const R  = MatrixX::Random(n, n);
    CSCMatrix A      = (R.transpose() * R).sparseView();
    A                = A.triangularView<Eigen::Lower>();
    math::linalg::Cholmod LLT{};
    SUBCASE("Can solve SPD linear systems")
    {
        LLT.analyze(A);
        bool const bFactorized = LLT.factorize();
        CHECK(bFactorized);
        auto constexpr m        = 3;
        MatrixX const X         = MatrixX::Random(n, m);
        MatrixX const B         = A.selfadjointView<Eigen::Lower>() * X;
        MatrixX const Xcomputed = LLT.solve(B);
        Scalar const error      = (X - Xcomputed).squaredNorm();
        auto constexpr zero     = 1e-15;
        CHECK_LE(error, zero);

        VectorX const u = VectorX::Random(n);
        A += (u * u.transpose()).sparseView();
        bool const bFactorized2 = LLT.compute(A);
        CHECK(bFactorized2);
        MatrixX const B2         = A.selfadjointView<Eigen::Lower>() * X;
        MatrixX const Xcomputed2 = LLT.solve(B2);
        Scalar const error2      = (X - Xcomputed2).squaredNorm();
        CHECK_LE(error2, zero);
    }
}
