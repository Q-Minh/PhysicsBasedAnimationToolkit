// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "LinearSolver.cuh"

namespace pbat::gpu::impl::math {

LinearSolver::LinearSolver() : mIsInitialized(false)
{
    cusolverStatus_t status = cusolverDnCreate(&mCusolverHandle);
    mIsInitialized          = (status == cusolverStatus_t::CUSOLVER_STATUS_SUCCESS);
}

bool LinearSolver::IsInitialized() const
{
    return mIsInitialized;
}

LinearSolver::~LinearSolver()
{
    if (mIsInitialized)
    {
        cusolverDnDestroy(mCusolverHandle);
    }
}

} // namespace pbat::gpu::impl::math
