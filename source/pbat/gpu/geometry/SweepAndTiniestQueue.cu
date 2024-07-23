#include "SweepAndTiniestQueue.cuh"

namespace pbat {
namespace gpu {
namespace geometry {

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>
#include <thrust/async/copy.h>
#include <thrust/async/for_each.h>
#include <thrust/host_vector.h>

TEST_CASE("[gpu] Sweep and tiniest queue")
{
    thrust::device_vector<int> d_inds(10);
    thrust::device_event e = thrust::async::copy(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(10),
        d_inds.begin());
    e.wait();
    thrust::host_vector<int> h_inds = d_inds;
    for (int i = 0; i < 10; ++i)
    {
        CHECK_EQ(h_inds[i], i);
    }
}