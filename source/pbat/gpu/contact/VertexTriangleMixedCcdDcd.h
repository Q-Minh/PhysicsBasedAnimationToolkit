/**
 * @file VertexTriangleMixedCcdDcd.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the VertexTriangleMixedCcdDcd class for vertex-triangle mixed
 * continuous collision detection (CCD) and discrete collision detection (DCD) on the GPU.
 * @date 2025-03-25
 *
 * @copyright Copyright (c) 2025
 *
 */

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

/**
 * @brief Public API of vertex-triangle mixed continuous collision detection (CCD) and discrete
 * collision detection system on the GPU.
 */
class VertexTriangleMixedCcdDcd
{
  public:
    static auto constexpr kDims = 3; ///< Number of spatial dimensions

    /**
     * @brief Construct a new Vertex Triangle Mixed Ccd Dcd object
     * @param B Body map s.t. `B(i) = j` means vertex `i` belongs to body `j`
     * @param V `|V|` matrix of collision vertex indices
     * @param F `3 x |F|` matrix of collision triangles
     */
    PBAT_API VertexTriangleMixedCcdDcd(
        Eigen::Ref<GpuIndexVectorX const> const& B,
        Eigen::Ref<GpuIndexVectorX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& F);

    VertexTriangleMixedCcdDcd(VertexTriangleMixedCcdDcd const&)            = delete;
    VertexTriangleMixedCcdDcd& operator=(VertexTriangleMixedCcdDcd const&) = delete;

    /**
     * @brief Move constructor
     * @param other VertexTriangleMixedCcdDcd to move from
     */
    PBAT_API VertexTriangleMixedCcdDcd(VertexTriangleMixedCcdDcd&& other) noexcept;
    /**
     * @brief Move assignment operator
     * @param other VertexTriangleMixedCcdDcd to move from
     * @return Reference to this VertexTriangleMixedCcdDcd
     */
    PBAT_API VertexTriangleMixedCcdDcd& operator=(VertexTriangleMixedCcdDcd&& other) noexcept;
    /**
     * @brief Computes the initial active set.
     *
     * @param xt Vertex positions at time t
     * @param xtp1 Vertex positions at time t+1
     * @param wmin World box minimum
     * @param wmax World box maximum
     */
    PBAT_API void InitializeActiveSet(
        common::Buffer const& xt,
        common::Buffer const& xtp1,
        Eigen::Vector<GpuScalar, kDims> const& wmin,
        Eigen::Vector<GpuScalar, kDims> const& wmax);
    /**
     * @brief Updates constraints involved with active vertices.
     * @param x Current vertex positions
     * @param bComputeBoxes If true, computes the bounding boxes of (non-swept) triangles.
     */
    PBAT_API void UpdateActiveSet(common::Buffer const& x, bool bComputeBoxes = true);
    /**
     * @brief Removes inactive vertices from the active set.
     * @param x Current vertex positions
     * @param bComputeBoxes If true, computes the bounding boxes of (non-swept) triangles.
     */
    PBAT_API void FinalizeActiveSet(common::Buffer const& x, bool bComputeBoxes = true);
    /**
     * @brief Fetch the active vertex-triangle constraints from GPU.
     * @return `2x|# vertex-triangle constraints|` matrix where each column is a
     * vertex-triangle constraint.
     */
    PBAT_API GpuIndexMatrixX ActiveVertexTriangleConstraints() const;
    /**
     * @brief Fetch the active vertices from GPU.
     * @return `|# active verts|` vector of active vertices.
     */
    PBAT_API GpuIndexVectorX ActiveVertices() const;
    /**
     * @brief Fetch the active vertex mask from GPU.
     * @return `|# verts|` vector of active vertex mask.
     */
    PBAT_API std::vector<bool> ActiveMask() const;
    /**
     * @brief Set the Nearest Neighbour floating point equality tolerance.
     * @param eps Tolerance
     */
    PBAT_API void SetNearestNeighbourFloatingPointTolerance(GpuScalar eps);
    /**
     * @brief Destroy the Vertex Triangle Mixed Ccd Dcd object
     */
    ~VertexTriangleMixedCcdDcd();

  private:
    impl::contact::VertexTriangleMixedCcdDcd* mImpl; ///< Pointer to the implementation
};

} // namespace pbat::gpu::contact

#endif // PBAT_GPU_CONTACT_VERTEXTRIANGLEMIXEDCCDDCD_H
