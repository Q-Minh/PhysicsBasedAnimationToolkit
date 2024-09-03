// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "BvhImpl.cuh"

#include <cuda/atomic>
#include <cuda/std/cmath>
#include <exception>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <type_traits>

namespace pbat {
namespace gpu {
namespace geometry {

struct FComputeAabb
{
    __device__ void operator()(int s)
    {
        for (auto d = 0; d < 3; ++d)
        {
            auto bs  = leafBegin + s;
            b[d][bs] = x[d][inds[0][s]];
            e[d][bs] = x[d][inds[0][s]];
            for (auto m = 1; m < nSimplexVertices; ++m)
            {
                b[d][bs] = cuda::std::fminf(b[d][bs], x[d][inds[m][s]]);
                e[d][bs] = cuda::std::fmaxf(e[d][bs], x[d][inds[m][s]]);
            }
            b[d][bs] -= r;
            e[d][bs] += r;
        }
    }

    std::array<GpuScalar const*, 3> x;
    std::array<GpuIndex const*, 4> inds;
    int nSimplexVertices;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuIndex leafBegin;
    GpuScalar r;
};

struct FComputeMortonCode
{
    using MortonCodeType = cuda::std::uint32_t;

    // Expands a 10-bit integer into 30 bits
    // by inserting 2 zeros after each bit.
    __device__ MortonCodeType ExpandBits(MortonCodeType v)
    {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

    // Calculates a 30-bit Morton code for the
    // given 3D point located within the unit cube [0,1].
    __device__ MortonCodeType Morton3D(std::array<GpuScalar, 3> x)
    {
        using namespace cuda::std;
        x[0]        = fminf(fmaxf(x[0] * 1024.0f, 0.0f), 1023.0f);
        x[1]        = fminf(fmaxf(x[1] * 1024.0f, 0.0f), 1023.0f);
        x[2]        = fminf(fmaxf(x[2] * 1024.0f, 0.0f), 1023.0f);
        uint32_t xx = ExpandBits(static_cast<uint32_t>(x[0]));
        uint32_t yy = ExpandBits(static_cast<uint32_t>(x[1]));
        uint32_t zz = ExpandBits(static_cast<uint32_t>(x[2]));
        return xx * 4 + yy * 2 + zz;
    }

    __device__ void operator()(int s)
    {
        auto const bs = leafBegin + s;
        // Compute Morton code of the centroid of the bounding box of simplex s
        std::array<GpuScalar, 3> c{0.f, 0.f, 0.f};
        for (auto d = 0; d < 3; ++d)
            c[d] += GpuScalar{0.5} * (b[d][bs] + e[d][bs]);
        morton[s] = Morton3D(c);
    }

    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuIndex* morton;
    GpuIndex leafBegin;
};

struct FGenerateHierarchy
{
    struct Range
    {
        GpuIndex i, j, l;
        int d;
    };

    __device__ int Delta(GpuIndex i, GpuIndex j) const
    {
        if (j < 0 or j >= n)
            return -1;
        if (i == j)
            return __clz(reinterpret_cast<int>(i ^ j));
        return __clz(reinterpret_cast<int>(morton[i] ^ morton[j]));
    }

    __device__ Range DetermineRange(GpuIndex i) const
    {
        // Compute range direction
        int const d = (Delta(i, i + 1) - Delta(i, i - 1)) >= 0;
        // Lower bound on length of internal node i's common prefix
        int const dmin = Delta(i, i - d);
        // Compute conservative upper bound on the range's size
        GpuIndex lmax{2};
        while (Delta(i, i + lmax * d) > dmin)
            lmax <<= 1;
        // Binary search in the "inflated" range for the actual end (or start) of internal node i's
        // range, considering that i is its start (or end).
        GpuIndex l{0};
        do
        {
            lmax >>= 1;
            if (Delta(i, i + (l + lmax) * d) > dmin)
                l += lmax;
        } while (lmax > 1);
        GpuIndex j = i + l * d;
        return Range{i, j, l, d};
    }

    __device__ GpuIndex FindSplit(Range R) const
    {
        // Identical Morton codes => split the range in the middle.
        if (morton[R.i] == morton[R.j])
            return (R.i + R.j) >> 1;

        // Calculate the number of highest bits that are the same
        // for all objects.
        int const dnode = Delta(R.i, R.j);

        // Use binary search to find where the next bit differs.
        // Specifically, we are looking for the highest object that
        // shares more than dnode bits with the first one.
        GpuIndex s{0};
        do
        {
            R.l = (R.l + 1) >> 1;
            if (Delta(R.i, R.i + (s + R.l) * R.d) > dnode)
                s += R.l;
        } while (R.l > 1);
        GpuIndex const gamma = R.i + s * R.d + min(R.d, 0);
        return gamma;
    }

    __device__ void operator()(auto i)
    {
        // Find out which range of objects the node corresponds to.
        Range R = DetermineRange(i);
        // Determine where to split the range.
        GpuIndex gamma = FindSplit(R);
        // Select left+right child
        GpuIndex lc = (min(R.i, R.j) == gamma) ? leafBegin + gamma : gamma;
        GpuIndex rc = (max(R.i, R.j) == gamma + 1) ? leafBegin + gamma + 1 : gamma + 1;
        // Record parent-child relationships
        child[0][i] = lc;
        child[1][i] = rc;
        parent[lc]  = i;
        parent[rc]  = i;
    }

    GpuIndex const* morton;
    std::array<GpuIndex*, 2> child;
    GpuIndex* parent;
    GpuIndex leafBegin;
    std::size_t n;
};

void BvhImpl::Build(PointsImpl const& P, SimplicesImpl const& S, GpuScalar expansion)
{
    auto const n = S.NumberOfSimplices();
    if (NumberOfAllocatedBoxes() < n)
    {
        std::string const what = "Allocated memory for " +
                                 std::to_string(NumberOfAllocatedBoxes()) +
                                 " boxes, but received " + std::to_string(n) + " simplices.";
        throw std::invalid_argument(what);
    }

    // 1. Construct leaf node (i.e. simplex) bounding boxes
    auto const leafBegin = n - 1;
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
            leafBegin,
            expansion});

    // 2. Compute Morton codes for each leaf node (i.e. simplex)
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        FComputeMortonCode{b.Raw(), e.Raw(), morton.Raw(), leafBegin});

    // 3. Sort simplices based on Morton codes
    thrust::sequence(thrust::device, simplex.Data(), simplex.Data());
    auto zip = thrust::make_zip_iterator(
        b[0].begin() + leafBegin,
        b[1].begin() + leafBegin,
        b[2].begin() + leafBegin,
        e[0].begin() + leafBegin,
        e[1].begin() + leafBegin,
        e[2].begin() + leafBegin,
        simplex.Data());
    // Using a stable sort preserves the initial ordering of simplex indices 0...n-1, resulting in
    // simplices sorted by Morton codes first, and then by simplex index.
    thrust::stable_sort_by_key(thrust::device, morton.Data(), morton.Data() + n, zip);

    // 4. Construct hierarchy
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n - 1),
        FGenerateHierarchy{morton.Raw(), child.Raw(), parent.Raw(), leafBegin, n});

    // 5. Construct internal node bounding boxes
}

std::size_t BvhImpl::NumberOfAllocatedBoxes() const
{
    return simplex.Size();
}

} // namespace geometry
} // namespace gpu
} // namespace pbat