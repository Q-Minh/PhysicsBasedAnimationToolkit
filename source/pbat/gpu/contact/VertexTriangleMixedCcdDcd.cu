// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VertexTriangleMixedCcdDcd.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/common/Eigen.cuh"
#include "pbat/gpu/impl/common/SynchronizedList.cuh"
#include "pbat/gpu/impl/contact/VertexTriangleMixedCcdDcd.cuh"
#include "pbat/math/linalg/mini/Eigen.h"

#include <algorithm>
#include <exception>
#include <string>

namespace pbat::gpu::contact {

VertexTriangleMixedCcdDcd::VertexTriangleMixedCcdDcd(
    Eigen::Ref<GpuIndexVectorX const> const& B,
    Eigen::Ref<GpuIndexVectorX const> const& V,
    Eigen::Ref<GpuIndexMatrixX const> const& F)
    : mImpl(new impl::contact::VertexTriangleMixedCcdDcd(B, V, F))
{
}

VertexTriangleMixedCcdDcd::VertexTriangleMixedCcdDcd(VertexTriangleMixedCcdDcd&& other) noexcept
    : mImpl(other.mImpl)
{
    other.mImpl = nullptr;
}

VertexTriangleMixedCcdDcd&
VertexTriangleMixedCcdDcd::operator=(VertexTriangleMixedCcdDcd&& other) noexcept
{
    if (this != &other)
    {
        if (mImpl != nullptr)
            delete mImpl;
        mImpl       = other.mImpl;
        other.mImpl = nullptr;
    }
    return *this;
}

void VertexTriangleMixedCcdDcd::InitializeActiveSet(
    common::Buffer const& xt,
    common::Buffer const& xtp1,
    Eigen::Vector<GpuScalar, kDims> const& wmin,
    Eigen::Vector<GpuScalar, kDims> const& wmax)
{
    using pbat::math::linalg::mini::FromEigen;
    if (xt.Dims() != kDims || xtp1.Dims() != kDims || xt.Size() != xtp1.Size())
    {
        throw std::invalid_argument(
            "Expected x(t) and x(t+1) to have the same 3x|#points| dimensions, but got xt=" +
            std::to_string(xt.Dims()) + "x" + std::to_string(xt.Size()) + " and x(t+1)=" +
            std::to_string(xtp1.Dims()) + "x" + std::to_string(xtp1.Size()) + " instead.");
    }
    if (xt.Type() != common::Buffer::EType::float32 ||
        xtp1.Type() != common::Buffer::EType::float32)
    {
        throw std::invalid_argument("Expected x(t) and x(t+1) to be of type float32.");
    }
    auto const* xtImpl   = static_cast<impl::common::Buffer<GpuScalar, 3> const*>(xt.Impl());
    auto const* xtp1Impl = static_cast<impl::common::Buffer<GpuScalar, 3> const*>(xtp1.Impl());
    mImpl->InitializeActiveSet(*xtImpl, *xtp1Impl, FromEigen(wmin), FromEigen(wmax));
}

void VertexTriangleMixedCcdDcd::UpdateActiveSet(common::Buffer const& x, bool bComputeBoxes)
{
    if (x.Dims() != kDims || x.Type() != common::Buffer::EType::float32)
    {
        throw std::invalid_argument(
            "Expected x of type float32 with 3x|#points| dimensions, but got x=" +
            std::to_string(x.Dims()) + "x" + std::to_string(x.Size()) + " instead.");
    }
    auto const* xImpl = static_cast<impl::common::Buffer<GpuScalar, 3> const*>(x.Impl());
    mImpl->UpdateActiveSet(*xImpl, bComputeBoxes);
}

void VertexTriangleMixedCcdDcd::FinalizeActiveSet(common::Buffer const& x, bool bComputeBoxes)
{
    if (x.Dims() != kDims || x.Type() != common::Buffer::EType::float32)
    {
        throw std::invalid_argument(
            "Expected x of type float32 with 3x|#points| dimensions, but got x=" +
            std::to_string(x.Dims()) + "x" + std::to_string(x.Size()) + " instead.");
    }
    auto const* xImpl = static_cast<impl::common::Buffer<GpuScalar, 3> const*>(x.Impl());
    mImpl->FinalizeActiveSet(*xImpl, bComputeBoxes);
}

GpuIndexMatrixX VertexTriangleMixedCcdDcd::ActiveVertexTriangleConstraints() const
{
    auto constexpr kMaxNeighbours = impl::contact::VertexTriangleMixedCcdDcd::kMaxNeighbours;
    GpuIndexVectorX av            = ActiveVertices();
    auto nActive                  = av.size();
    GpuIndexMatrixX nn            = impl::common::ToEigen(mImpl->nn).reshaped(
        kMaxNeighbours,
        mImpl->nn.Size() / kMaxNeighbours);
    GpuIndexVectorX counts(nActive);
    counts.setZero();
    for (auto j = 0; j < nActive; ++j)
    {
        auto const v = av(j);
        for (auto k = 0; k < nn.rows() and nn(k, v) >= 0; ++k)
            ++counts(j);
    }
    auto const nConstraints = counts.sum();
    GpuIndexMatrixX A(2, nConstraints);
    for (auto j = 0, c = 0; j < nActive; ++j)
    {
        for (auto k = 0; k < counts(j); ++k, ++c)
        {
            auto v  = av(j);
            A(0, c) = v;
            A(1, c) = nn(k, v);
        }
    }
    return A;
}

GpuIndexVectorX VertexTriangleMixedCcdDcd::ActiveVertices() const
{
    auto av = impl::common::ToEigen(mImpl->av);
    return av.reshaped().head(mImpl->nActive);
}

std::vector<bool> VertexTriangleMixedCcdDcd::ActiveMask() const
{
    return mImpl->active.Get();
}

void VertexTriangleMixedCcdDcd::SetNearestNeighbourFloatingPointTolerance(GpuScalar eps)
{
    mImpl->eps = eps;
}

VertexTriangleMixedCcdDcd::~VertexTriangleMixedCcdDcd()
{
    if (mImpl != nullptr)
        delete mImpl;
}

} // namespace pbat::gpu::contact