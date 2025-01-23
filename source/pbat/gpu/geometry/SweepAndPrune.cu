// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "SweepAndPrune.h"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/common/SynchronizedList.cuh"
#include "pbat/gpu/impl/geometry/SweepAndPrune.cuh"

#include <cuda/std/utility>

namespace pbat {
namespace gpu {
namespace geometry {

SweepAndPrune::SweepAndPrune(std::size_t nPrimitives, std::size_t nOverlaps)
    : mImpl(new impl::geometry::SweepAndPrune(nPrimitives)),
      mOverlaps(new impl::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>(nOverlaps))
{
    static_assert(
        alignof(cuda::std::pair<GpuIndex, GpuIndex>) == sizeof(GpuIndex) and
            sizeof(cuda::std::pair<GpuIndex, GpuIndex>) == 2 * sizeof(GpuIndex),
        "Assumed that std::vector<cuda::std::pair<GpuIndex, GpuIndex>> is contiguous");
}

SweepAndPrune::SweepAndPrune(SweepAndPrune&& other) noexcept
    : mImpl(other.mImpl), mOverlaps(other.mOverlaps)
{
    other.mImpl     = nullptr;
    other.mOverlaps = nullptr;
}

SweepAndPrune& SweepAndPrune::operator=(SweepAndPrune&& other) noexcept
{
    if (this != &other)
    {
        Deallocate();
        mImpl           = other.mImpl;
        mOverlaps       = other.mOverlaps;
        other.mImpl     = nullptr;
        other.mOverlaps = nullptr;
    }
    return *this;
}

GpuIndexMatrixX SweepAndPrune::SortAndSweep(Aabb& aabbs)
{
    auto constexpr kDims = impl::geometry::SweepAndPrune::kDims;
    if (aabbs.Dimensions() != kDims)
    {
        throw std::invalid_argument(
            "Expected AABBs to have 3 dimensions, but got " + std::to_string(aabbs.Dimensions()));
    }
    auto* aabbImpl     = static_cast<impl::geometry::Aabb<kDims>*>(aabbs.Impl());
    using Overlap      = cuda::std::pair<GpuIndex, GpuIndex>;
    using OverlapPairs = impl::common::SynchronizedList<Overlap>;
    auto* overlaps     = static_cast<OverlapPairs*>(mOverlaps);
    overlaps->Clear();
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

PBAT_API GpuIndexMatrixX SweepAndPrune::SortAndSweep(common::Buffer const& set, Aabb& aabbs)
{
    auto constexpr kDims = impl::geometry::SweepAndPrune::kDims;
    if (aabbs.Dimensions() != kDims)
    {
        throw std::invalid_argument(
            "Expected AABBs to have 3 dimensions, but got " + std::to_string(aabbs.Dimensions()));
    }
    if (set.Size() != aabbs.Size() and set.Dims() != 1 and
        set.Type() != common::Buffer::EType::int32)
    {
        throw std::invalid_argument(
            "Expected set to have the same size as AABBs and be a 1D vector of type int32");
    }
    auto* aabbImpl      = static_cast<impl::geometry::Aabb<kDims>*>(aabbs.Impl());
    auto const* setImpl = static_cast<impl::common::Buffer<GpuIndex> const*>(set.Impl());
    // NOTE:
    // Unfortunately, we have to allocate on-the-fly here.
    // We should define a type-erased CPU wrapper over gpu::common::Buffer to prevent this.
    using Overlap      = cuda::std::pair<GpuIndex, GpuIndex>;
    using OverlapPairs = impl::common::SynchronizedList<Overlap>;
    auto* overlaps     = static_cast<OverlapPairs*>(mOverlaps);
    overlaps->Clear();
    mImpl->SortAndSweep(
        *aabbImpl,
        [S = setImpl->Raw(), o = overlaps->Raw()] PBAT_DEVICE(GpuIndex si, GpuIndex sj) mutable {
            if (S[si] != S[sj])
                o.Append(cuda::std::make_pair(si, sj));
        });
    auto O         = overlaps->Get();
    auto nOverlaps = static_cast<GpuIndex>(O.size());
    GpuIndex* data = reinterpret_cast<GpuIndex*>(std::addressof(O.front()));
    return Eigen::Map<GpuIndexMatrixX>(data, 2, nOverlaps);
}

SweepAndPrune::~SweepAndPrune()
{
    Deallocate();
}

void SweepAndPrune::Deallocate()
{
    if (mImpl != nullptr)
    {
        delete mImpl;
    }
    if (mOverlaps != nullptr)
    {
        delete static_cast<impl::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>*>(
            mOverlaps);
    }
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[gpu][geometry] SweepAndPrune")
{
    using namespace pbat;
    // Arrange
    GpuMatrixX V(3, 7);
    GpuIndexMatrixX E1(2, 3);
    GpuIndexMatrixX F2(3, 1);
    // clang-format off
    V << 0.f,  1.f ,  2.f ,  3.f , 0.f,  2.f ,  0.f,
         0.f,  0.1f,  0.2f,  0.3f, 0.f,  0.1f,  0.f,
         0.f, 10.f , 20.f , 30.f , 0.f, 10.f ,  0.f;
    E1 << 1, 0, 2,
          2, 1, 3;
    F2 << 4,
          5,
          6;
    // clang-format on
    using gpu::geometry::Aabb;
    auto const nEdges     = static_cast<GpuIndex>(E1.cols());
    auto const nTriangles = static_cast<GpuIndex>(F2.cols());
    // NOTE:
    // Maybe we should factor this code out into some reusable function that computes lower/upper
    // endpoints for simplex sets.
    GpuMatrixX L(3, nEdges + nTriangles);
    GpuMatrixX U(3, nEdges + nTriangles);
    GpuMatrixX VE(V.rows() * nEdges, E1.rows());
    for (auto d = 0; d < E1.rows(); ++d)
        VE.col(d) = V(Eigen::placeholders::all, E1.row(d)).reshaped();
    GpuMatrixX VF(V.rows() * nTriangles, F2.rows());
    for (auto d = 0; d < F2.rows(); ++d)
        VF.col(d) = V(Eigen::placeholders::all, F2.row(d)).reshaped();
    L.leftCols(nEdges)      = VE.rowwise().minCoeff().reshaped(3, nEdges);
    U.leftCols(nEdges)      = VE.rowwise().maxCoeff().reshaped(3, nEdges);
    L.rightCols(nTriangles) = VF.rowwise().minCoeff().reshaped(3, nTriangles);
    U.rightCols(nTriangles) = VF.rowwise().maxCoeff().reshaped(3, nTriangles);
    Aabb aabbs(3, nEdges + nTriangles);
    aabbs.Construct(L, U);
    GpuIndexMatrixX Oexpected(2, 2);
    // clang-format off
    Oexpected << nEdges+0, nEdges+0, /*face 0 = nEdges+0*/
                 1, 0;
    // clang-format on
    // Act
    gpu::geometry::SweepAndPrune sap(nEdges + nTriangles, 4);
    GpuIndexVectorX set(nEdges + nTriangles);
    set.segment(0, nEdges).setZero();
    set.segment(nEdges, nTriangles).setOnes();
    gpu::common::Buffer setBuffer(set);
    GpuIndexMatrixX O = sap.SortAndSweep(setBuffer, aabbs);
    // Assert
    CHECK_EQ(O.rows(), 2);
    CHECK_EQ(O.cols(), Oexpected.cols());
    auto detected = GpuIndexVectorX::Zero(Oexpected.cols()).eval();
    for (auto o = 0; o < O.cols(); ++o)
    {
        // Find overlap in expected overlaps
        for (auto oe = 0; oe < Oexpected.cols(); ++oe)
            if ((O(0, o) == Oexpected(0, oe) and O(1, o) == Oexpected(1, oe)) or
                (O(0, o) == Oexpected(1, oe) and O(1, o) == Oexpected(0, oe)))
                ++detected(oe);
    }
    bool const bAllOverlapsDetectedOnce = (detected.array() == GpuIndex(1)).all();
    CHECK(bAllOverlapsDetectedOnce);
}