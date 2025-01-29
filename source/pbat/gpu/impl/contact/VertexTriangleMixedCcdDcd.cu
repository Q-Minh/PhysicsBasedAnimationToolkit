// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "VertexTriangleMixedCcdDcd.cuh"

#include <thrust/sequence.h>

namespace pbat::gpu::impl::contact {

VertexTriangleMixedCcdDcd::VertexTriangleMixedCcdDcd(GpuIndex nVerts, GpuIndex nTriangles)
    : inds(nVerts),
      morton(nVerts),
      Paabbs(nVerts),
      Faabbs(nTriangles),
      Fbvh(nTriangles),
      active(nVerts),
      av(nVerts),
      nn(nVerts * kMaxNeighbours),
      sd(nVerts)
{
}

void VertexTriangleMixedCcdDcd::InitializeActiveSet(
    [[maybe_unused]] common::Buffer<GpuScalar, 3> const& xt,
    [[maybe_unused]] common::Buffer<GpuScalar, 3> const& xtp1,
    [[maybe_unused]] common::Buffer<GpuIndex> const& V,
    [[maybe_unused]] common::Buffer<GpuIndex, 3> const& F)
{
    // TODO: Implement
}

void VertexTriangleMixedCcdDcd::UpdateActiveSet(
    [[maybe_unused]] common::Buffer<GpuScalar, 3> const& x)
{
    // TODO: Implement
}

void VertexTriangleMixedCcdDcd::FinalizeActiveSet(
    [[maybe_unused]] common::Buffer<GpuScalar, 3> const& x)
{
    // TODO: Implement
}

} // namespace pbat::gpu::impl::contact