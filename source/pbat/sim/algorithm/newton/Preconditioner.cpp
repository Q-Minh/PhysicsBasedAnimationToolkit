#include "Preconditioner.h"

namespace pbat::sim::algorithm::newton {

PreconditionerOperator::PreconditionerOperator([[maybe_unused]] HessianOperator const& A)
    : mImpl(nullptr)
{
    // no-op
}

PreconditionerOperator& PreconditionerOperator::analyzePattern(HessianOperator const& A)
{
    if (not mImpl->mIsPatternAnalyzed)
    {
        mImpl->mLLT.analyzePattern(A.mData->HNC);
        mImpl->mIsPatternAnalyzed = true;
    }
    return *this;
}

PreconditionerOperator& PreconditionerOperator::factorize(HessianOperator const& A)
{
    mImpl->mLLT.factorize(A.mData->HNC);
    return *this;
}

PreconditionerOperator& PreconditionerOperator::compute(HessianOperator const& A)
{
    analyzePattern(A);
    factorize(A);
    return *this;
}

Eigen::ComputationInfo PreconditionerOperator::info() const
{
    return mImpl->mLLT.info();
}

} // namespace pbat::sim::algorithm::newton