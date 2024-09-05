#include "BvhQueryImpl.cuh"

#include <cuda/atomic>
#include <cuda/std/cmath>
#include <exception>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pbat {
namespace gpu {
namespace geometry {

BvhQueryImpl::BvhQueryImpl(std::size_t nPrimitives, std::size_t nOverlaps)
    : simplex(nPrimitives), morton(nPrimitives), b(nPrimitives), e(nPrimitives), no(0), o(nOverlaps)
{
}

struct FComputeAabb
{
    __device__ void operator()(int s)
    {
        for (auto d = 0; d < 3; ++d)
        {
            b[d][s] = x[d][inds[0][s]];
            e[d][s] = x[d][inds[0][s]];
            for (auto m = 1; m < nSimplexVertices; ++m)
            {
                b[d][s] = cuda::std::fminf(b[d][s], x[d][inds[m][s]]);
                e[d][s] = cuda::std::fmaxf(e[d][s], x[d][inds[m][s]]);
            }
            b[d][s] -= r;
            e[d][s] += r;
        }
    }

    std::array<GpuScalar const*, 3> x;
    std::array<GpuIndex const*, 4> inds;
    int nSimplexVertices;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuScalar r;
};

struct FComputeMortonCode
{
    using MortonCodeType = typename BvhImpl::MortonCodeType;

    __device__ void operator()(int s)
    {
        // Compute Morton code of the centroid of the bounding box of simplex s
        std::array<GpuScalar, 3> c{0.f, 0.f, 0.f};
        for (auto d = 0; d < 3; ++d)
            c[d] += GpuScalar{0.5} * (b[d][s] + e[d][s]);
        morton[s] = common::Morton3D(c);
    }

    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    MortonCodeType* morton;
};

void BvhQueryImpl::Build(PointsImpl const& P, SimplicesImpl const& S, GpuScalar expansion)
{
    auto const n = S.NumberOfSimplices();
    if (NumberOfAllocatedBoxes() < n)
    {
        std::string const what = "Allocated memory for " +
                                 std::to_string(NumberOfAllocatedBoxes()) +
                                 " boxes, but received " + std::to_string(n) + " simplices.";
        throw std::invalid_argument(what);
    }
    // Compute bounding boxes
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        FComputeAabb{
            P.x.Raw(),
            S.inds.Raw(),
            static_cast<int>(S.eSimplexType),
            b.Raw(),
            e.Raw(),
            expansion});
    // Compute simplex morton codes
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        FComputeMortonCode{b.Raw(), e.Raw(), morton.Raw()});
    // Sort simplices+boxes by morton codes to try and improve data locality in future queries
    thrust::sequence(thrust::device, simplex.Data(), simplex.Data() + simplex.Size());
    auto zip = thrust::make_zip_iterator(
        b[0].begin(),
        b[1].begin(),
        b[2].begin(),
        e[0].begin(),
        e[1].begin(),
        e[2].begin(),
        simplex.Data());
    thrust::stable_sort_by_key(thrust::device, morton.Data(), morton.Data() + n, zip);
}

void BvhQueryImpl::DetectOverlaps(
    PointsImpl const& P1,
    SimplicesImpl const& S1,
    PointsImpl const& P2,
    SimplicesImpl const& S2)
{
}

std::size_t BvhQueryImpl::NumberOfAllocatedBoxes() const
{
    return simplex.Size();
}

} // namespace geometry
} // namespace gpu
} // namespace pbat