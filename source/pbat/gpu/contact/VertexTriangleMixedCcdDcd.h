#ifndef PBAT_GPU_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H
#define PBAT_GPU_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.h"

#include <vector>

namespace pbat::gpu::impl::contact {
class VertexTriangleMixedCcdDcd;
} // namespace pbat::gpu::impl::contact

namespace pbat::gpu::contact {

class VertexTriangleMixedCcdDcd
{
  public:
    static auto constexpr kDims = 3;

    /**
     * @brief Construct a new Vertex Triangle Mixed Ccd Dcd object
     *
     * @param B
     * @param V
     * @param F
     */
    PBAT_API VertexTriangleMixedCcdDcd(
        Eigen::Ref<GpuIndexVectorX const> const& B,
        Eigen::Ref<GpuIndexVectorX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& F);

    VertexTriangleMixedCcdDcd(VertexTriangleMixedCcdDcd const&)            = delete;
    VertexTriangleMixedCcdDcd& operator=(VertexTriangleMixedCcdDcd const&) = delete;

    PBAT_API VertexTriangleMixedCcdDcd(VertexTriangleMixedCcdDcd&& other) noexcept;
    PBAT_API VertexTriangleMixedCcdDcd& operator=(VertexTriangleMixedCcdDcd&& other) noexcept;

    /**
     * @brief Computes the initial active set.
     *
     * @param xt
     * @param xtp1
     * @param wmin
     * @param wmax
     */
    PBAT_API void InitializeActiveSet(
        common::Buffer const& xt,
        common::Buffer const& xtp1,
        Eigen::Vector<GpuScalar, kDims> const& wmin,
        Eigen::Vector<GpuScalar, kDims> const& wmax);
    /**
     * @brief Updates constraints involved with active vertices.
     *
     * @param x
     * @param bComputeBoxes If true, computes the bounding boxes of (non-swept) triangles.
     */
    PBAT_API void UpdateActiveSet(common::Buffer const& x, bool bComputeBoxes = true);
    /**
     * @brief Removes inactive vertices from the active set.
     *
     * @param x
     * @param bComputeBoxes If true, computes the bounding boxes of (non-swept) triangles.
     */
    PBAT_API void FinalizeActiveSet(common::Buffer const& x, bool bComputeBoxes = true);
    /**
     * @brief
     *
     * @return GpuIndexMatrixX 2x|#vertex-triangle constraints| matrix where each column is a
     * vertex-triangle constraint.
     */
    PBAT_API GpuIndexMatrixX ActiveVertexTriangleConstraints() const;
    /**
     * @brief
     *
     * @return GpuIndexVectorX
     */
    PBAT_API GpuIndexVectorX ActiveVertices() const;
    /**
     * @brief
     *
     * @return PBAT_API
     */
    PBAT_API std::vector<bool> ActiveMask() const;
    /**
     * @brief Set the Nearest Neighbour floating point equality tolerance.
     *
     * @param eps
     */
    PBAT_API void SetNearestNeighbourFloatingPointTolerance(GpuScalar eps);
    /**
     * @brief Destroy the Vertex Triangle Mixed Ccd Dcd object
     *
     */
    ~VertexTriangleMixedCcdDcd();

  private:
    impl::contact::VertexTriangleMixedCcdDcd* mImpl;
};

} // namespace pbat::gpu::contact

#endif // PBAT_GPU_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H
