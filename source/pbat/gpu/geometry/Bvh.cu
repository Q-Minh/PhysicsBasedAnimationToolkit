// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Bvh.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/common/SynchronizedList.cuh"
#include "pbat/gpu/impl/geometry/Bvh.cuh"
#include "pbat/math/linalg/mini/Eigen.h"

#include <cuda/std/utility>
#include <exception>
#include <string>

namespace pbat {
namespace gpu {
namespace geometry {

Bvh::Bvh(GpuIndex nPrimitives, [[maybe_unused]] GpuIndex nOverlaps)
    : mImpl(new impl::geometry::Bvh(nPrimitives)),
      mOverlaps(new impl::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>(nOverlaps))
{
    static_assert(
        alignof(cuda::std::pair<GpuIndex, GpuIndex>) == sizeof(GpuIndex) and
            sizeof(cuda::std::pair<GpuIndex, GpuIndex>) == 2 * sizeof(GpuIndex),
        "Assumed that std::vector<cuda::std::pair<GpuIndex, GpuIndex>> is contiguous");
}

Bvh::Bvh(Bvh&& other) noexcept : mImpl(other.mImpl), mOverlaps(other.mOverlaps)
{
    other.mImpl     = nullptr;
    other.mOverlaps = nullptr;
}

Bvh& Bvh::operator=(Bvh&& other) noexcept
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

void Bvh::Build(
    Aabb& aabbs,
    Eigen::Vector<GpuScalar, 3> const& min,
    Eigen::Vector<GpuScalar, 3> const& max)
{
    using namespace pbat::math::linalg;
    if (aabbs.Dimensions() != impl::geometry::Bvh::kDims)
    {
        throw std::invalid_argument(
            "Expected AABBs to have " + std::to_string(impl::geometry::Bvh::kDims) +
            " dimensions, but got " + std::to_string(aabbs.Dimensions()));
    }
    mImpl->Build(
        *static_cast<impl::geometry::Aabb<impl::geometry::Bvh::kDims>*>(aabbs.Impl()),
        mini::FromEigen(min),
        mini::FromEigen(max));
}

GpuIndexMatrixX Bvh::DetectOverlaps(Aabb const& aabbs)
{
    if (aabbs.Dimensions() != impl::geometry::Bvh::kDims)
    {
        throw std::invalid_argument(
            "Expected AABBs to have " + std::to_string(impl::geometry::Bvh::kDims) +
            " dimensions, but got " + std::to_string(aabbs.Dimensions()));
    }
    auto overlaps =
        static_cast<impl::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>*>(
            mOverlaps);
    overlaps->Clear();
    mImpl->DetectOverlaps(
        *static_cast<impl::geometry::Aabb<impl::geometry::Bvh::kDims>*>(aabbs.Impl()),
        [o = overlaps->Raw()] PBAT_DEVICE(GpuIndex si, GpuIndex sj) mutable {
            o.Append(cuda::std::make_pair(si, sj));
        });
    auto O         = overlaps->Get();
    auto nOverlaps = static_cast<GpuIndex>(O.size());
    GpuIndex* data = reinterpret_cast<GpuIndex*>(std::addressof(O.front()));
    return Eigen::Map<GpuIndexMatrixX>(data, 2, nOverlaps);
}

GpuIndexMatrixX Bvh::DetectOverlaps(common::Buffer const& set, Aabb const& aabbs)
{
    if (aabbs.Dimensions() != impl::geometry::Bvh::kDims)
    {
        throw std::invalid_argument(
            "Expected AABBs to have " + std::to_string(impl::geometry::Bvh::kDims) +
            " dimensions, but got " + std::to_string(aabbs.Dimensions()));
    }
    if (set.Size() != aabbs.Size() and set.Dims() != 1 and
        set.Type() != common::Buffer::EType::int32)
    {
        throw std::invalid_argument(
            "Expected set to have the same size as AABBs and be a 1D vector of type int32");
    }
    auto* aabbImpl = static_cast<impl::geometry::Aabb<impl::geometry::Bvh::kDims>*>(aabbs.Impl());
    auto const* setImpl = static_cast<impl::common::Buffer<GpuIndex> const*>(set.Impl());
    auto overlaps =
        static_cast<impl::common::SynchronizedList<cuda::std::pair<GpuIndex, GpuIndex>>*>(
            mOverlaps);
    overlaps->Clear();
    mImpl->DetectOverlaps(
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

GpuMatrixX Bvh::Min() const
{
    return impl::common::ToEigen(mImpl->iaabbs.b);
}

GpuMatrixX Bvh::Max() const
{
    return impl::common::ToEigen(mImpl->iaabbs.e);
}

GpuIndexVectorX Bvh::LeafOrdering() const
{
    return impl::common::ToEigen(mImpl->inds);
}

Eigen::Vector<typename Bvh::MortonCodeType, Eigen::Dynamic> Bvh::MortonCodes() const
{
    return impl::common::ToEigen(mImpl->morton.codes);
}

GpuIndexMatrixX Bvh::Child() const
{
    return impl::common::ToEigen(mImpl->child).transpose();
}

GpuIndexVectorX Bvh::Parent() const
{
    return impl::common::ToEigen(mImpl->parent);
}

GpuIndexMatrixX Bvh::Rightmost() const
{
    return impl::common::ToEigen(mImpl->rightmost).transpose();
}

GpuIndexVectorX Bvh::Visits() const
{
    return impl::common::ToEigen(mImpl->visits);
}

Bvh::~Bvh()
{
    Deallocate();
}

impl::geometry::Bvh* Bvh::Impl()
{
    return mImpl;
}

impl::geometry::Bvh const* Bvh::Impl() const
{
    return mImpl;
}

void Bvh::Deallocate()
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

TEST_CASE("[gpu][geometry] Bvh")
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
    auto wmin = L.rowwise().minCoeff();
    auto wmax = U.rowwise().maxCoeff();
    GpuIndexVectorX set(nEdges + nTriangles);
    set.segment(0, nEdges).setZero();
    set.segment(nEdges, nTriangles).setOnes();
    gpu::common::Buffer setBuffer(set);

    // Act
    gpu::geometry::Bvh bvh(nEdges + nTriangles, 4);
    bvh.Build(aabbs, wmin, wmax);
    GpuIndexMatrixX O = bvh.DetectOverlaps(setBuffer, aabbs);

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