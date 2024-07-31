#include "SweepAndPruneImpl.cuh"

#include <cuda/atomic>
#include <cuda/std/cmath>
#include <exception>
#include <string>
#include <thrust/async/copy.h>
#include <thrust/async/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace pbat {
namespace gpu {
namespace geometry {

SweepAndPruneImpl::SweepAndPruneImpl(std::size_t nPrimitives, std::size_t nOverlaps)
    : binds(nPrimitives),
      b({thrust::device_vector<GpuScalar>(nPrimitives),
         thrust::device_vector<GpuScalar>(nPrimitives),
         thrust::device_vector<GpuScalar>(nPrimitives)}),
      e({thrust::device_vector<GpuScalar>(nPrimitives),
         thrust::device_vector<GpuScalar>(nPrimitives),
         thrust::device_vector<GpuScalar>(nPrimitives)}),
      mu(3, 0.f),
      sigma(3, 0.f),
      no(1),
      o(nOverlaps)
{
}

struct FComputeAabb
{
    FComputeAabb(
        std::array<thrust::device_ptr<GpuScalar const>, 3> xIn,
        Simplices const& SIn,
        std::array<thrust::device_ptr<GpuScalar>, 3> bIn,
        std::array<thrust::device_ptr<GpuScalar>, 3> eIn)
        : x({thrust::raw_pointer_cast(xIn[0]),
             thrust::raw_pointer_cast(xIn[1]),
             thrust::raw_pointer_cast(xIn[2])}),
          nSimplexVertices(static_cast<int>(SIn.eSimplexType)),
          inds(thrust::raw_pointer_cast(SIn.inds.data())),
          b({thrust::raw_pointer_cast(bIn[0]),
             thrust::raw_pointer_cast(bIn[1]),
             thrust::raw_pointer_cast(bIn[2])}),
          e({thrust::raw_pointer_cast(eIn[0]),
             thrust::raw_pointer_cast(eIn[1]),
             thrust::raw_pointer_cast(eIn[2])})
    {
    }

    __device__ void operator()(int s)
    {
        auto const begin = s * nSimplexVertices;
        auto const end   = begin + nSimplexVertices;
        for (auto d = 0; d < 3; ++d)
        {
            b[d][s] = x[d][inds[begin]];
            e[d][s] = x[d][inds[begin]];
            for (auto i = begin + 1; i < end; ++i)
            {
                b[d][s] = cuda::std::fminf(b[d][s], x[d][inds[i]]);
                e[d][s] = cuda::std::fmaxf(b[d][s], x[d][inds[i]]);
            }
            // TODO: Add some inflation to bounding box endpoints to support activation distance
        }
    }

    std::array<GpuScalar const*, 3> x;
    int nSimplexVertices;
    GpuIndex const* inds;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
};

struct FComputeMean
{
    FComputeMean(
        std::array<thrust::device_ptr<GpuScalar const>, 3> bIn,
        std::array<thrust::device_ptr<GpuScalar const>, 3> eIn,
        thrust::device_ptr<GpuScalar> muIn,
        GpuIndex nBoxesIn)
        : b({thrust::raw_pointer_cast(bIn[0]),
             thrust::raw_pointer_cast(bIn[1]),
             thrust::raw_pointer_cast(bIn[2])}),
          e({thrust::raw_pointer_cast(eIn[0]),
             thrust::raw_pointer_cast(eIn[1]),
             thrust::raw_pointer_cast(eIn[2])}),
          mu(thrust::raw_pointer_cast(muIn)),
          nBoxes(nBoxesIn)
    {
    }

    __device__ void operator()(int s)
    {
        cuda::atomic_ref<GpuScalar, cuda::thread_scope_device> amu[3] = {mu[0], mu[1], mu[2]};
        for (auto d = 0; d < 3; ++d)
        {
            amu[d] += (b[d][s] + e[d][s]) / (2.f * static_cast<GpuScalar>(nBoxes));
        }
    }

    std::array<GpuScalar const*, 3> b;
    std::array<GpuScalar const*, 3> e;
    GpuScalar* mu;
    GpuIndex nBoxes;
};

struct FComputeVariance
{
    FComputeVariance(
        std::array<thrust::device_ptr<GpuScalar const>, 3> bIn,
        std::array<thrust::device_ptr<GpuScalar const>, 3> eIn,
        thrust::device_ptr<GpuScalar> muIn,
        thrust::device_ptr<GpuScalar> sigmaIn,
        GpuIndex nBoxesIn)
        : b({thrust::raw_pointer_cast(bIn[0]),
             thrust::raw_pointer_cast(bIn[1]),
             thrust::raw_pointer_cast(bIn[2])}),
          e({thrust::raw_pointer_cast(eIn[0]),
             thrust::raw_pointer_cast(eIn[1]),
             thrust::raw_pointer_cast(eIn[2])}),
          mu(muIn),
          sigma(thrust::raw_pointer_cast(sigmaIn)),
          nBoxes(nBoxesIn)
    {
    }

    __device__ void operator()(int s)
    {
        cuda::atomic_ref<GpuScalar, cuda::thread_scope_device> asigma[3] = {
            sigma[0],
            sigma[1],
            sigma[2]};
        for (auto d = 0; d < 3; ++d)
        {
            GpuScalar const cd = (b[d][s] + e[d][s]) / 2.f;
            GpuScalar const dx = cd - mu[d];
            asigma[d] += dx * dx / static_cast<GpuScalar>(nBoxes);
        }
    }

    std::array<GpuScalar const*, 3> b;
    std::array<GpuScalar const*, 3> e;
    thrust::device_ptr<GpuScalar const> mu;
    GpuScalar* sigma;
    GpuIndex nBoxes;
};

__device__ bool AreSimplicesAdjacent(
    GpuIndex* sinds1,
    int nSimplexVertices1,
    GpuIndex s1,
    GpuIndex* sinds2,
    int nSimplexVertices2,
    GpuIndex s2)
{
    auto const begin1 = s1 * nSimplexVertices1;
    auto const end1   = begin1 + nSimplexVertices1;
    auto const begin2 = s2 * nSimplexVertices2;
    auto const end2   = begin2 + nSimplexVertices2;
    bool bAreAdjacent{false};
    for (auto i = begin1; (i < end1) and not bAreAdjacent; ++i)
    {
        for (auto j = begin2; j < end2; ++j)
        {
            if (sinds1[i] == sinds2[j])
            {
                bAreAdjacent = true;
                break;
            }
        }
    }
    return bAreAdjacent;
}

//__global__ void SweepImpl(
//    GpuIndex* binds,
//    GpuScalar* b[3],
//    GpuScalar* e[3],
//    GpuIndex nSimplices1,
//    GpuIndex* sinds1,
//    int nSimplexVertices1,
//    GpuIndex* sinds2,
//    int nSimplexVertices2,
//    GpuIndex saxis,
//    GpuIndex axis[2],
//    GpuIndex nBoxes,
//    GpuIndex* no,
//    cuda::std::pair<GpuIndex, GpuIndex>* o)
//{
//    cuda::atomic_ref<GpuIndex, cuda::thread_scope_device> ano{*no};
//    GpuIndex const t = threadIdx.x + blockIdx.x * blockDim.x;
//    if (t >= nBoxes)
//        return;
//
//    GpuIndex tp = t + 1;
//    while (tp >= nBoxes and e[saxis][t] > b[saxis][tp])
//    {
//        GpuIndex const s1                         = binds[t];
//        bool const bAreSimplicesFromDifferentSets = binds[tp] > nSimplices1;
//        if (bAreSimplicesFromDifferentSets)
//        {
//            GpuIndex const s2 =
//                bAreSimplicesFromDifferentSets ? binds[tp] - nSimplices1 : binds[tp];
//            bool const bAreAdjacent =
//                AreSimplicesAdjacent(sinds1, nSimplexVertices1, s1, sinds2, nSimplexVertices2,
//                s2);
//            bool const bAreOverlapping =
//                e[axis[0]][t] > b[axis[0]][tp] and e[axis[1]][t] > b[axis[1]][tp];
//            if (bAreOverlapping and not bAreAdjacent)
//            {
//                GpuIndex const oid = ano++;
//                o[oid]             = {s1, s2};
//            }
//        }
//        ++tp;
//    }
//}

void SweepAndPruneImpl::SortAndSweep(Points const& P, Simplices const& S1, Simplices const& S2)
{
    auto const nSimplices1 = S1.NumberOfSimplices();
    auto const nSimplices2 = S2.NumberOfSimplices();
    auto const nBoxes      = nSimplices1 + nSimplices2;
    if (NumberOfAllocatedBoxes() < nBoxes)
    {
        std::string const what = "Allocated memory for " +
                                 std::to_string(NumberOfAllocatedBoxes()) +
                                 " boxes, but received " + std::to_string(nBoxes) + " simplices.";
        throw std::invalid_argument(what);
    }

    // 1. Compute bounding boxes of S1 and S2
    thrust::device_event computeAabbEvent1 = thrust::async::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nSimplices1),
        FComputeAabb(
            {P.x.data(), P.y.data(), P.z.data()},
            S1,
            {b[0].data(), b[1].data(), b[2].data()},
            {e[0].data(), e[1].data(), e[2].data()}));
    thrust::device_event computeAabbEvent2 = thrust::async::for_each(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(nSimplices2),
        FComputeAabb(
            {P.x.data(), P.y.data(), P.z.data()},
            S2,
            {b[0].data() + nSimplices1, b[1].data() + nSimplices1, b[2].data() + nSimplices1},
            {e[0].data() + nSimplices1, e[1].data() + nSimplices1, e[2].data() + nSimplices1}));

    // 2. Compute mean and variance of bounding box centers
    auto const boxesBegin = thrust::make_counting_iterator(0);
    auto const boxesEnd   = thrust::make_counting_iterator(nBoxes);
    thrust::fill(mu.begin(), mu.end(), 0.f);
    thrust::fill(sigma.begin(), sigma.end(), 0.f);
    FComputeMean fComputeMean(
        {b[0].data(), b[1].data(), b[2].data()},
        {e[0].data(), e[1].data(), e[2].data()},
        mu.data(),
        nBoxes);
    thrust::device_event computeMeanEvent = thrust::async::for_each(
        thrust::device.after(computeAabbEvent1, computeAabbEvent2),
        boxesBegin,
        boxesEnd,
        fComputeMean);
    FComputeVariance fComputeVariance(
        {b[0].data(), b[1].data(), b[2].data()},
        {e[0].data(), e[1].data(), e[2].data()},
        mu.data(),
        sigma.data(),
        nBoxes);
    thrust::device_event computeVarianceEvent = thrust::async::for_each(
        thrust::device.after(computeMeanEvent),
        boxesBegin,
        boxesEnd,
        fComputeVariance);
    computeVarianceEvent.wait();

    // 3. Sort bounding boxes along largest variance axis
    GpuIndex const saxis =
        (sigma[0] > sigma[1]) ? (sigma[0] > sigma[2] ? 0 : 2) : (sigma[1] > sigma[2] ? 1 : 2);
    thrust::sequence(thrust::device, binds.begin(), binds.end());
    GpuIndex const axis[2] = {(saxis + 1) % 3, (saxis + 2) % 3};
    thrust::sort_by_key(
        thrust::device,
        b[saxis].begin(),
        b[saxis].end(),
        thrust::make_zip_iterator(
            binds.begin(),
            b[axis[0]].begin(),
            b[axis[1]].begin(),
            e[axis[0]].begin(),
            e[axis[1]].begin()));

    // 4. Sweep to find overlaps
    thrust::fill(no.begin(), no.end(), 0);
    auto const nThreadsPerBlock = 32;
    auto const nBlocks          = (nBoxes - 1) / nThreadsPerBlock + 1;
    // SweepImpl<<<nBlocks, nThreadsPerBlock>>>();
    //  TODO: ...
}

std::size_t SweepAndPruneImpl::NumberOfAllocatedBoxes() const
{
    return binds.size();
}

std::size_t SweepAndPruneImpl::NumberOfAllocatedOverlaps() const
{
    return o.size();
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[gpu][geometry] Sweep and tiniest queue")
{
    using namespace pbat;
    MatrixX V(3, 4);
    // clang-format off
    V << 0., 1., 2., 3.,
         0., 0., 0., 0.,
         0., 10., 20., 30.;
    // clang-format on
    IndexMatrixX E(2, 3);
    // clang-format off
    E << 1, 0, 2,
         2, 1, 3;
    // clang-format on
    gpu::geometry::Points P(V);
    gpu::geometry::Simplices S(E);

    gpu::geometry::SweepAndPruneImpl stq(3, 2);
    // stq.SortAndSweep(P, S);
}
