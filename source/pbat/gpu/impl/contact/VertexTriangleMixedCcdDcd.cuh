#ifndef PBAT_GPU_IMPL_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H
#define PBAT_GPU_IMPL_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/geometry/Aabb.cuh"
#include "pbat/gpu/impl/geometry/Bvh.cuh"
#include "pbat/gpu/impl/geometry/Morton.cuh"

namespace pbat::gpu::impl::contact {

class VertexTriangleMixedCcdDcd
{
  public:
    static auto constexpr kDims          = 3;
    static auto constexpr kMaxNeighbours = 8;

    /**
     * @brief Construct a new Vertex Triangle Mixed Ccd Dcd object
     *
     * @param V
     * @param F
     */
    VertexTriangleMixedCcdDcd(
        common::Buffer<GpuIndex> const& V,
        common::Buffer<GpuIndex, 3> const& F);
    /**
     * @brief Computes the initial active set.
     *
     * Let VA denote the set of vertices whose line segments (xt -> xtp1) overlap with swept
     * triangles (xt_F -> xtp1_F), combined with active vertices from the previous time step.
     *
     * The initial active set is computed as all pairs (i, f) such that i in VA and f are nearest
     * neighbour triangles to i.
     *
     * @param xt
     * @param xtp1
     * @param wmin
     * @param wmax
     */
    void InitializeActiveSet(
        common::Buffer<GpuScalar, 3> const& xt,
        common::Buffer<GpuScalar, 3> const& xtp1,
        geometry::Morton::Bound const& wmin,
        geometry::Morton::Bound const& wmax);
    /**
     * @brief The active set is updated by recomputing nearest neighbours f of i, using current
     * signed distances sd(i,f) as a warm-start.
     *
     * @param x
     */
    void UpdateActiveSet(common::Buffer<GpuScalar, 3> const& x);
    /**
     * @brief Finalizes the active set by removing all pairs (i, f) such that sd(i,f) > 0.
     *
     * @param x
     */
    void FinalizeActiveSet(common::Buffer<GpuScalar, 3> const& x);

  private:
    common::Buffer<GpuIndex> V;    ///< Vertices
    common::Buffer<GpuIndex, 3> F; ///< Triangles
    common::Buffer<Index> inds;    ///< |#pts| point indices i
    geometry::Morton morton;       ///< |#pts| morton codes for points
    geometry::Aabb<kDims>
        Paabbs; ///< |#pts| axis-aligned bounding boxes of swept points (i.e. line segments)
    geometry::Aabb<kDims>
        Faabbs;         ///< |#tris| axis-aligned bounding boxes of (potentially swept) triangles
    geometry::Bvh Fbvh; ///< Bounding volume hierarchy over (potentially swept) triangles
    common::Buffer<bool> active; ///< |#pts| active mask
    common::Buffer<Index> av;    ///< Active vertices (i.e. indices of active points)
    GpuIndex nActive;            ///< Number of active vertices
    common::Buffer<Index> nn;    ///< |#pts*kMaxNeighbours| nearest neighbours f to pts i.
                                 ///< nn[i*kMaxNeighbours+j] < 0 if no neighbour
    common::Buffer<Scalar> sd;   ///< |#pts| signed distance min_f sd(i,f) to surface
};

} // namespace pbat::gpu::impl::contact

#endif // PBAT_GPU_IMPL_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H
