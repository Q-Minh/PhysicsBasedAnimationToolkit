#include "SweepAndTiniestQueue.cuh"

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/execution_policy.h>
#include <thrust/uninitialized_fill.h>

namespace pbat {
namespace gpu {
namespace geometry {

SweepAndTiniestQueue::SweepAndTiniestQueue(std::size_t nPrimitives, std::size_t nOverlaps)
    : inds(thrust::device_malloc<GpuIndex>(nPrimitives)),
      bx(thrust::device_malloc<GpuScalar>(nPrimitives)),
      by(thrust::device_malloc<GpuScalar>(nPrimitives)),
      bz(thrust::device_malloc<GpuScalar>(nPrimitives)),
      ex(thrust::device_malloc<GpuScalar>(nPrimitives)),
      ey(thrust::device_malloc<GpuScalar>(nPrimitives)),
      ez(thrust::device_malloc<GpuScalar>(nPrimitives)),
      mux(thrust::device_malloc<AtomicScalarType>(1)),
      muy(thrust::device_malloc<AtomicScalarType>(1)),
      muz(thrust::device_malloc<AtomicScalarType>(1)),
      sigmax(thrust::device_malloc<AtomicScalarType>(1)),
      sigmay(thrust::device_malloc<AtomicScalarType>(1)),
      sigmaz(thrust::device_malloc<AtomicScalarType>(1)),
      no(thrust::device_malloc<AtomicSizeType>(1)),
      o(thrust::device_malloc<OverlapType>(nOverlaps))
{
    thrust::uninitialized_fill(thrust::device, mux, mux + 1, 0.f);
    thrust::uninitialized_fill(thrust::device, muy, muy + 1, 0.f);
    thrust::uninitialized_fill(thrust::device, muz, muz + 1, 0.f);
    thrust::uninitialized_fill(thrust::device, sigmax, sigmax + 1, 0.f);
    thrust::uninitialized_fill(thrust::device, sigmay, sigmay + 1, 0.f);
    thrust::uninitialized_fill(thrust::device, sigmaz, sigmaz + 1, 0.f);
    thrust::uninitialized_fill(thrust::device, no, no + 1, 0);
}

SweepAndTiniestQueue::~SweepAndTiniestQueue()
{
    thrust::device_free(inds);
    thrust::device_free(bx);
    thrust::device_free(by);
    thrust::device_free(bz);
    thrust::device_free(ex);
    thrust::device_free(ey);
    thrust::device_free(ez);
    thrust::device_free(mux);
    thrust::device_free(muy);
    thrust::device_free(muz);
    thrust::device_free(sigmax);
    thrust::device_free(sigmay);
    thrust::device_free(sigmaz);
    thrust::device_free(no);
    thrust::device_free(o);
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <cuda/std/atomic>
#include <doctest/doctest.h>
#include <thrust/async/copy.h>
#include <thrust/async/for_each.h>
#include <thrust/async/reduce.h>
#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/host_vector.h>

struct CountOp
{
    using AtomicSizeType = cuda::atomic<int, cuda::thread_scope_device>;

    __host__ __device__ CountOp(thrust::device_ptr<AtomicSizeType> ptr) : ac(ptr) {}
    __host__ __device__ CountOp(CountOp const& other) : ac(other.ac) {}
    __host__ __device__ CountOp& operator=(CountOp const& other)
    {
        ac = other.ac;
        return *this;
    }

    __host__ __device__ void operator()(int i) { ac->fetch_add(i); }

    thrust::device_ptr<AtomicSizeType> ac;
};

struct TransformOp
{
    using AtomicSizeType = cuda::atomic<int, cuda::thread_scope_device>;

    __host__ __device__ int operator()(AtomicSizeType const& ac) const { return ac.load(); }
};

TEST_CASE("[gpu] Sweep and tiniest queue")
{
    thrust::device_vector<int> d_inds(10);
    thrust::device_event eSequence = thrust::async::copy(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(10),
        d_inds.begin());
    eSequence.wait();
    thrust::host_vector<int> h_inds = d_inds;
    for (int i = 0; i < 10; ++i)
    {
        CHECK_EQ(h_inds[i], i);
    }

    using AtomicSizeType = typename CountOp::AtomicSizeType;

    thrust::device_ptr<AtomicSizeType> countPtr = thrust::device_malloc<AtomicSizeType>(1);
    thrust::uninitialized_fill(thrust::device, countPtr, countPtr + 1, 0);
    CountOp op{countPtr};
    thrust::device_event eCount =
        thrust::async::for_each(thrust::device, d_inds.begin(), d_inds.end(), op);
    eCount.wait();
    AtomicSizeType ac;
    cudaMemcpy(&ac, countPtr.get(), sizeof(AtomicSizeType), cudaMemcpyDeviceToHost);
    int count = ac.load();
    thrust::device_free(countPtr);
    CHECK_EQ(count, 45);
}
