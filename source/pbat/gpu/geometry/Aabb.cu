// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "Aabb.h"
#include "impl/Aabb.cuh"
#include "pbat/common/ConstexprFor.h"
#include "pbat/common/Eigen.h"
#include "pbat/gpu/common/Eigen.cuh"

#include <exception>
#include <string>

namespace pbat {
namespace gpu {
namespace geometry {

Aabb::Aabb(GpuIndex dims, GpuIndex nPrimitives) : mDims(dims), mImpl(nullptr)
{
    Resize(dims, nPrimitives);
}

Aabb::Aabb(Aabb&& other) noexcept : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

PBAT_API Aabb& Aabb::operator=(Aabb&& other) noexcept
{
    Deallocate();
    mImpl       = other.mImpl;
    other.mImpl = nullptr;
    return *this;
}

void Aabb::Construct(Eigen::Ref<GpuMatrixX const> const& L, Eigen::Ref<GpuMatrixX const> const& U)
{
    if (L.rows() != U.rows() or L.cols() != U.cols())
    {
        throw std::invalid_argument("Expected L and U to have the same shape.");
    }
    Resize(L.rows(), L.cols());
    pbat::common::ForRange<1, 4>([&]<auto kDims>() {
        if (mDims == kDims)
        {
            auto* impl = static_cast<impl::Aabb<kDims>*>(mImpl);
            gpu::common::ToBuffer(L, impl->b);
            gpu::common::ToBuffer(U, impl->e);
        }
    });
}

void Aabb::Resize(GpuIndex dims, GpuIndex nPrimitives)
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
                static_cast<impl::Aabb<kDims>*>(mImpl)->Resize(nPrimitives);
        });
    }
    else
    {
        pbat::common::ForRange<1, 4>([&]<auto kDims>() {
            if (mDims == kDims)
                mImpl = new impl::Aabb<kDims>(nPrimitives);
        });
    }
}

std::size_t Aabb::Size() const
{
    if (not mImpl)
        return 0;
    std::size_t size;
    pbat::common::ForRange<1, 4>([&]<auto kDims>() {
        if (mDims == kDims)
            size = static_cast<impl::Aabb<kDims>*>(mImpl)->Size();
    });
    return size;
}

GpuMatrixX Aabb::Lower() const
{
    GpuMatrixX L;
    pbat::common::ForRange<1, 4>([&]<auto kDims>() {
        if (mImpl and mDims == kDims)
        {
            auto* impl = static_cast<impl::Aabb<kDims>*>(mImpl);
            L = common::ToEigen(impl->b.Get()).reshaped(impl->b.Dimensions(), impl->b.Size());
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
            auto* impl = static_cast<impl::Aabb<kDims>*>(mImpl);
            U = common::ToEigen(impl->e.Get()).reshaped(impl->e.Dimensions(), impl->e.Size());
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
            delete static_cast<impl::Aabb<kDims>*>(mImpl);
            mImpl = nullptr;
        }
    });
}

} // namespace geometry
} // namespace gpu
} // namespace pbat