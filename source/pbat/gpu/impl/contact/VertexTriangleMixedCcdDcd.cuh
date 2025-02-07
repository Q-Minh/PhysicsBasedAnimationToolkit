#ifndef PBAT_GPU_IMPL_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H
#define PBAT_GPU_IMPL_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H

#include "pbat/geometry/DistanceQueries.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/geometry/Aabb.cuh"
#include "pbat/gpu/impl/geometry/Bvh.cuh"
#include "pbat/gpu/impl/geometry/Morton.cuh"
#include "pbat/math/linalg/mini/Mini.h"

namespace pbat::gpu::impl::contact {

class VertexTriangleMixedCcdDcd
{
  public:
    static auto constexpr kDims          = 3;
    static auto constexpr kMaxNeighbours = 8;

    /**
     * @brief Construct a new Vertex Triangle Mixed Ccd Dcd object
     *
     * @param B
     * @param V
     * @param F
     */
    VertexTriangleMixedCcdDcd(
        Eigen::Ref<GpuIndexVectorX const> const& B,
        Eigen::Ref<GpuIndexVectorX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& F);
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
     * distances d(i,f) as a warm-start.
     *
     * @param x
     * @param bComputeBoxes If true, recomputes the AABBs of the (non-swept) triangles.
     */
    void UpdateActiveSet(common::Buffer<GpuScalar, 3> const& x, bool bComputeBoxes = true);
    /**
     * @brief Finalizes the active set by removing all pairs (i, f) such that sd(i,f) > 0.
     *
     * @param x
     * @param bComputeBoxes If true, recomputes the AABBs of the (non-swept) triangles.
     */
    void FinalizeActiveSet(common::Buffer<GpuScalar, 3> const& x, bool bComputeBoxes = true);
    /**
     * @brief
     *
     * @tparam FOnNearestNeighbourFound
     * @param x
     * @param fOnNearestNeighbourFound
     */
    template <class FOnNearestNeighbourFound>
    void ForEachNearestNeighbour(
        common::Buffer<GpuScalar, 3> const& x,
        FOnNearestNeighbourFound fOnNearestNeighbourFound);
    /**
     * @brief
     *
     * @param x
     */
    void UpdateBvh(common::Buffer<GpuScalar, 3> const& x);

  public:
    common::Buffer<GpuIndex> av;   ///< Active vertices
    GpuIndex nActive;              ///< Number of active vertices
    common::Buffer<GpuIndex> nn;   ///< |#verts*kMaxNeighbours| nearest neighbours f to vertices v.
                                   ///< nn[v*kMaxNeighbours+j] < 0 if no neighbour
    common::Buffer<GpuIndex> B;    ///< |#pts| body map
    common::Buffer<GpuIndex> V;    ///< Vertices
    common::Buffer<GpuIndex, 3> F; ///< Triangles

    common::Buffer<GpuIndex> inds; ///< |#verts| vertex indices v
    geometry::Morton morton;       ///< |#verts| morton codes for vertices
    geometry::Aabb<kDims>
        Paabbs; ///< |#verts| axis-aligned bounding boxes of swept vertices (i.e. line segments)
    geometry::Aabb<kDims>
        Faabbs;         ///< |#tris| axis-aligned bounding boxes of (potentially swept) triangles
    geometry::Bvh Fbvh; ///< Bounding volume hierarchy over (potentially swept) triangles
    common::Buffer<bool> active;      ///< |#verts| active mask
    common::Buffer<GpuScalar> dupper; ///< |#verts| NN search radius
    GpuScalar eps;                    ///< Tolerance for NN searches
};

template <class FOnNearestNeighbourFound>
inline void VertexTriangleMixedCcdDcd::ForEachNearestNeighbour(
    common::Buffer<GpuScalar, 3> const& x,
    FOnNearestNeighbourFound fOnNearestNeighbourFound)
{
    using namespace pbat::math::linalg::mini;
    auto fGetQueryObject = [x = x.Raw(), V = V.Raw(), av = av.Raw()] PBAT_DEVICE(GpuIndex q) {
        GpuIndex const v = av[q];
        return FromBuffers<3, 1>(x, V[v]);
    };
    auto fMinDistanceToBox = [] PBAT_DEVICE(
                                 SVector<GpuScalar, 3> const& xi,
                                 SVector<GpuScalar, 3> const& L,
                                 SVector<GpuScalar, 3> const& U) {
        return pbat::geometry::DistanceQueries::PointAxisAlignedBoundingBox(xi, L, U);
    };
    auto fDistanceToLeaf =
        [x = x.Raw(), V = V.Raw(), F = F.Raw(), B = B.Raw(), av = av.Raw()] PBAT_DEVICE(
            GpuIndex q,
            SVector<GpuScalar, 3> const& xi,
            [[maybe_unused]] GpuIndex leaf,
            GpuIndex f) {
            GpuIndex const i         = V[av[q]];
            auto fv                  = FromBuffers<3, 1>(F, f);
            bool const bFromSameBody = B[i] == B[fv[0]];
            if (bFromSameBody)
                return std::numeric_limits<GpuScalar>::max();

            SMatrix<GpuScalar, 3, 3> const xf = FromBuffers(x, fv.Transpose());
            return pbat::geometry::DistanceQueries::PointTriangle(
                xi,
                xf.Col(0),
                xf.Col(1),
                xf.Col(2));
        };
    auto fDistanceUpperBound = [dupper = dupper.Raw(), av = av.Raw()] PBAT_DEVICE(GpuIndex q) {
        // Try warm-starting the NN search!
        return dupper[av[q]];
    };
    Fbvh.NearestNeighbours<
        decltype(fGetQueryObject),
        decltype(fMinDistanceToBox),
        decltype(fDistanceToLeaf),
        decltype(fDistanceUpperBound),
        decltype(fOnNearestNeighbourFound),
        kMaxNeighbours>(
        Faabbs,
        nActive,
        fGetQueryObject,
        fMinDistanceToBox,
        fDistanceToLeaf,
        fDistanceUpperBound,
        fOnNearestNeighbourFound,
        eps);
}

} // namespace pbat::gpu::impl::contact

#endif // PBAT_GPU_IMPL_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H
