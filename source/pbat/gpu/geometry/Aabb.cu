// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Aabb.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/Eigen.h"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/geometry/Aabb.cuh"

#include <exception>
#include <string>

namespace pbat {
namespace gpu {
namespace geometry {

Aabb::Aabb(GpuIndex dims, GpuIndex nBoxes) : mDims(dims), mImpl(nullptr)
{
    Resize(dims, nBoxes);
}

Aabb::Aabb(Aabb&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

PBAT_API Aabb& Aabb::operator=(Aabb&& other) noexcept
{
    if (this != &other)
    {
        Deallocate();
        mImpl       = other.mImpl;
        other.mImpl = nullptr;
    }
    return *this;
}

void Aabb::Construct(Eigen::Ref<GpuMatrixX const> const& L, Eigen::Ref<GpuMatrixX const> const& U)
{
    if (L.rows() != U.rows() or L.cols() != U.cols())
    {
        throw std::invalid_argument("Expected L and U to have the same shape.");
    }
    Resize(static_cast<GpuIndex>(L.rows()), static_cast<GpuIndex>(L.cols()));
    pbat::common::ForRange<1, 4>([&]<auto kDims>() {
        if (mDims == kDims)
        {
            auto* impl = static_cast<impl::geometry::Aabb<kDims>*>(mImpl);
            impl::common::ToBuffer(L, impl->b);
            impl::common::ToBuffer(U, impl->e);
        }
    });
}

void Aabb::Construct(
    Eigen::Ref<GpuMatrixX const> const& P,
    Eigen::Ref<GpuIndexMatrixX const> const& S)
{
    if (P.rows() != 3)
    {
        throw std::invalid_argument(
            "Expected P to have 3 rows, but got " + std::to_string(P.rows()));
    }
    auto constexpr kDims    = 3;
    auto const nSimplices   = static_cast<GpuIndex>(S.cols());
    auto const nSimplexDims = static_cast<GpuIndex>(S.rows());
    Resize(kDims, nSimplices);
    pbat::common::ForRange<2, 5>([&]<auto kSimplexDims>() {
        if (nSimplexDims == kSimplexDims)
        {
            auto* impl = static_cast<impl::geometry::Aabb<kDims>*>(mImpl);
            impl::common::Buffer<GpuScalar, kDims> PG(P.cols());
            impl::common::ToBuffer(P, PG);
            impl::common::Buffer<GpuIndex, kSimplexDims> SG(S.cols());
            impl::common::ToBuffer(S, SG);
            impl->Construct<kSimplexDims>(PG, SG);
        }
    });
}

void Aabb::Resize(GpuIndex dims, GpuIndex nBoxes)
{
    if (dims < 1 or dims > 3)
    {
        throw std::invalid_argument(
            "Expected 1 <= dims <= 3, but received " + std::to_string(dims));
    }
    if (dims != mDims)
    {
        Deallocate();
    }
    mDims = dims;
    if (mImpl)
    {
        pbat::common::ForRange<1, 4>([&]<auto kDims>() {
            if (mDims == kDims)
                static_cast<impl::geometry::Aabb<kDims>*>(mImpl)->Resize(nBoxes);
        });
    }
    else
    {
        pbat::common::ForRange<1, 4>([&]<auto kDims>() {
            if (mDims == kDims)
                mImpl = new impl::geometry::Aabb<kDims>(nBoxes);
        });
    }
}

GpuIndex Aabb::Size() const
{
    if (not mImpl)
        return 0;
    std::size_t size;
    pbat::common::ForRange<1, 4>([&]<auto kDims>() {
        if (mDims == kDims)
            size = static_cast<impl::geometry::Aabb<kDims>*>(mImpl)->Size();
    });
    return static_cast<GpuIndex>(size);
}

GpuMatrixX Aabb::Lower() const
{
    GpuMatrixX L;
    pbat::common::ForRange<1, 4>([&]<auto kDims>() {
        if (mImpl and mDims == kDims)
        {
            auto* impl = static_cast<impl::geometry::Aabb<kDims>*>(mImpl);
            L          = impl::common::ToEigen(impl->b);
        }
    });
    return L;
}

GpuMatrixX Aabb::Upper() const
{
    GpuMatrixX U;
    pbat::common::ForRange<1, 4>([&]<auto kDims>() {
        if (mImpl and mDims == kDims)
        {
            auto* impl = static_cast<impl::geometry::Aabb<kDims>*>(mImpl);
            U          = impl::common::ToEigen(impl->e);
        }
    });
    return U;
}

Aabb::~Aabb()
{
    Deallocate();
}

void Aabb::Deallocate()
{
    pbat::common::ForRange<1, 4>([&]<auto kDims>() {
        if (mImpl and mDims == kDims)
        {
            delete static_cast<impl::geometry::Aabb<kDims>*>(mImpl);
            mImpl = nullptr;
        }
    });
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[gpu][geometry] Aabb")
{
    using namespace pbat;
    // Arrange
    // Cube mesh
    GpuMatrixX P(3, 8);
    GpuIndexMatrixX T(4, 5);
    // clang-format off
    P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    auto const dims  = static_cast<GpuIndex>(P.rows());
    auto const nTets = static_cast<GpuIndex>(T.cols());
    GpuMatrixX VT(dims * nTets, T.rows());
    for (auto d = 0; d < T.rows(); ++d)
        VT.col(d) = P(Eigen::placeholders::all, T.row(d)).reshaped();
    auto Lexpected = VT.rowwise().minCoeff().reshaped(dims, nTets).eval();
    auto Uexpected = VT.rowwise().maxCoeff().reshaped(dims, nTets).eval();

    // Act
    using gpu::geometry::Aabb;
    Aabb aabb(1, 8);
    CHECK_EQ(aabb.Dimensions(), 1);
    CHECK_EQ(aabb.Size(), 8);
    CHECK_NE(aabb.Impl(), nullptr);
    SUBCASE("Resize")
    {
        aabb.Resize(dims, nTets);
        CHECK_EQ(aabb.Dimensions(), dims);
        CHECK_EQ(aabb.Size(), nTets);
        CHECK_NE(aabb.Impl(), nullptr);
        SUBCASE("Construct")
        {
            aabb.Construct(Lexpected, Uexpected);
            auto L              = aabb.Lower();
            auto U              = aabb.Upper();
            bool const bIsEqual = L.isApprox(Lexpected) and U.isApprox(Uexpected);
            CHECK(bIsEqual);
        }
    }
}