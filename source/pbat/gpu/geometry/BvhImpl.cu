// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "BvhImpl.cuh"

#include <cuda/atomic>
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

struct FLeafBoundingBoxes
{
    __device__ void operator()(int s)
    {
        using namespace cuda::std;
        for (auto d = 0; d < 3; ++d)
        {
            auto bs  = leafBegin + s;
            b[d][bs] = x[d][inds[0][s]];
            e[d][bs] = x[d][inds[0][s]];
            for (auto m = 1; m < nSimplexVertices; ++m)
            {
                b[d][bs] = fminf(b[d][bs], x[d][inds[m][s]]);
                e[d][bs] = fmaxf(e[d][bs], x[d][inds[m][s]]);
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
    using MortonCodeType = typename BvhImpl::MortonCodeType;

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
        x[0]              = fminf(fmaxf(x[0] * 1024.0f, 0.0f), 1023.0f);
        x[1]              = fminf(fmaxf(x[1] * 1024.0f, 0.0f), 1023.0f);
        x[2]              = fminf(fmaxf(x[2] * 1024.0f, 0.0f), 1023.0f);
        MortonCodeType xx = ExpandBits(static_cast<MortonCodeType>(x[0]));
        MortonCodeType yy = ExpandBits(static_cast<MortonCodeType>(x[1]));
        MortonCodeType zz = ExpandBits(static_cast<MortonCodeType>(x[2]));
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
    MortonCodeType* morton;
    GpuIndex leafBegin;
};

struct FGenerateHierarchy
{
    using MortonCodeType = typename BvhImpl::MortonCodeType;

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
            return __clz(i ^ j);
        return __clz(morton[i] ^ morton[j]);
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

    MortonCodeType const* morton;
    std::array<GpuIndex*, 2> child;
    GpuIndex* parent;
    GpuIndex leafBegin;
    GpuIndex n;
};

struct FInternalNodeBoundingBoxes
{
    __device__ void operator()(auto leaf)
    {
        using namespace cuda::std;
        auto p = parent[leaf];
        while (p >= 0)
        {
            cuda::atomic_ref<GpuIndex, cuda::thread_scope_device> ap{visits[p]};
            // The first thread that gets access to the internal node p will terminate,
            // while the second thread visiting p will be allowed to continue execution.
            // This ensures that there is no race condition where a thread can access an
            // internal node too early, i.e. before both children of the internal node
            // have finished computing their bounding boxes.
            if (ap++ == 0)
                break;

            GpuIndex lc = child[0][p];
            GpuIndex rc = child[1][p];
            for (auto d = 0; d < 3; ++d)
            {
                b[d][p] = fminf(b[d][lc], b[d][rc]);
                e[d][p] = fmaxf(e[d][lc], e[d][rc]);
            }
            // Move up the binary tree
            p = parent[p];
        }
    }

    GpuIndex const* parent;
    std::array<GpuIndex*, 2> child;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuIndex* visits;
};

BvhImpl::BvhImpl(std::size_t nPrimitives, std::size_t nOverlaps)
    : simplex(nPrimitives),
      morton(nPrimitives),
      child(nPrimitives - 1),
      parent(2 * nPrimitives - 1),
      b(2 * nPrimitives - 1),
      e(2 * nPrimitives - 1),
      visits(nPrimitives - 1),
      no(0),
      o(nOverlaps)
{
    thrust::fill(thrust::device, parent.Data(), parent.Data() + parent.Size(), GpuIndex{-1});
}

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

    // 0. Reset intermediate data
    thrust::fill(thrust::device, visits.Raw(), visits.Raw() + visits.Size(), GpuIndex{0});

    // 1. Construct leaf node (i.e. simplex) bounding boxes
    auto const leafBegin = n - 1;
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(n),
        FLeafBoundingBoxes{
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
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator(n - 1),
        thrust::make_counting_iterator(2 * n - 1),
        FInternalNodeBoundingBoxes{parent.Raw(), child.Raw(), b.Raw(), e.Raw(), visits.Raw()});
}

std::size_t BvhImpl::NumberOfAllocatedBoxes() const
{
    return simplex.Size();
}

} // namespace geometry
} // namespace gpu
} // namespace pbat