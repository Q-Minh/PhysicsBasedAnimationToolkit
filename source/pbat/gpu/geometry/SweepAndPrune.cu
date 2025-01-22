// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "SweepAndPrune.h"
#include "impl/SweepAndPrune.cuh"
#include "pbat/gpu/common/SynchronizedList.cuh"

#include <cuda/std/utility>

namespace pbat {
namespace gpu {
namespace geometry {

SweepAndPrune::SweepAndPrune(std::size_t nPrimitives, std::size_t nOverlaps)
    : mImpl(new impl::SweepAndPrune(nPrimitives)),
      mOverlaps(new common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>(nOverlaps))
{
    static_assert(
        alignof(cuda::std::pair<GpuIndex, GpuIndex>) == sizeof(GpuIndex) and
            sizeof(cuda::std::pair<GpuIndex, GpuIndex>) == 2 * sizeof(GpuIndex),
        "Assumed that std::vector<cuda::std::pair<GpuIndex, GpuIndex>> is contiguous");
}

SweepAndPrune::SweepAndPrune(SweepAndPrune&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

SweepAndPrune& SweepAndPrune::operator=(SweepAndPrune&& other) noexcept
{
    if (mImpl != nullptr)
        delete mImpl;
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

GpuIndexMatrixX SweepAndPrune::SortAndSweep(Aabb& aabbs)
{
    auto constexpr kDims = impl::SweepAndPrune::kDims;
    if (aabbs.Dimensions() != kDims)
    {
        throw std::invalid_argument(
            "Expected AABBs to have 3 dimensions, but got " + std::to_string(aabbs.Dimensions()));
    }
    auto* aabbImpl     = static_cast<impl::Aabb<kDims>*>(aabbs.Impl());
    using Overlap      = cuda::std::pair<GpuIndex, GpuIndex>;
    using OverlapPairs = common::SynchronizedList<Overlap>;
    auto* overlaps     = static_cast<OverlapPairs*>(mOverlaps);
    mImpl->SortAndSweep(
        *aabbImpl,
        [o = overlaps->Raw()] PBAT_DEVICE(GpuIndex si, GpuIndex sj) mutable {
            o.Append(cuda::std::make_pair(si, sj));
        });
    auto O         = overlaps->Get();
    auto nOverlaps = static_cast<GpuIndex>(O.size());
    GpuIndex* data = reinterpret_cast<GpuIndex*>(std::addressof(O.front()));
    return Eigen::Map<GpuIndexMatrixX>(data, 2, nOverlaps);
}

PBAT_API GpuIndexMatrixX SweepAndPrune::SortAndSweep(GpuIndex n, Aabb& aabbs)
{
    auto constexpr kDims = impl::SweepAndPrune::kDims;
    if (aabbs.Dimensions() != kDims)
    {
        throw std::invalid_argument(
            "Expected AABBs to have 3 dimensions, but got " + std::to_string(aabbs.Dimensions()));
    }
    auto* aabbImpl     = static_cast<impl::Aabb<kDims>*>(aabbs.Impl());
    using Overlap      = cuda::std::pair<GpuIndex, GpuIndex>;
    using OverlapPairs = common::SynchronizedList<Overlap>;
    auto* overlaps     = static_cast<OverlapPairs*>(mOverlaps);
    mImpl->SortAndSweep(
        *aabbImpl,
        [n, o = overlaps->Raw()] PBAT_DEVICE(GpuIndex si, GpuIndex sj) mutable {
            if (si < n and sj >= n)
                o.Append(cuda::std::make_pair(si, sj - n));
            if (si >= n and sj < n)
                o.Append(cuda::std::make_pair(sj, si - n));
        });
    auto O         = overlaps->Get();
    auto nOverlaps = static_cast<GpuIndex>(O.size());
    GpuIndex* data = reinterpret_cast<GpuIndex*>(std::addressof(O.front()));
    return Eigen::Map<GpuIndexMatrixX>(data, 2, nOverlaps);
}

SweepAndPrune::~SweepAndPrune()
{
    if (mImpl != nullptr)
    {
        delete mImpl;
    }
    if (mOverlaps != nullptr)
    {
        delete static_cast<common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>*>(
            mOverlaps);
    }
}

} // namespace geometry
} // namespace gpu
} // namespace pbat