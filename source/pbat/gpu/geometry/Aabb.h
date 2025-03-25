/**
 * @file Aabb.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Axis-aligned bounding box (AABB) buffer on the GPU
 * @date 2025-03-25
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_GEOMETRY_AABB_H
#define PBAT_GPU_GEOMETRY_AABB_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace geometry {

/**
 * @brief GPU axis-aligned bounding box (AABB) buffer public API
 */
class Aabb
{
  public:
    /**
     * @brief Construct a new Aabb object
     *
     * @param dims Embedding dimensionality > 0
     * @param nBoxes
     */
    PBAT_API Aabb(GpuIndex dims = 3, GpuIndex nBoxes = 0);

    Aabb(Aabb const&)            = delete;
    Aabb& operator=(Aabb const&) = delete;
    /**
     * @brief Move constructor
     * @param other Aabb to move from
     */
    PBAT_API Aabb(Aabb&& other) noexcept;
    /**
     * @brief Move assignment operator
     * @param other Aabb to move from
     * @return Reference to this
     */
    PBAT_API Aabb& operator=(Aabb&& other) noexcept;
    /**
     * @brief Construct a new Aabb object from the lower and upper bounds matrices
     * @param L `dims x |# pts|` array of lower bounds
     * @param U `dims x |# pts|` array of upper bounds
     */
    PBAT_API void
    Construct(Eigen::Ref<GpuMatrixX const> const& L, Eigen::Ref<GpuMatrixX const> const& U);
    /**
     * @brief Construct a new Aabb object from shared vertex simplex mes (P,S)
     * @param P `dims x |# pts|` array of points
     * @param S `K x |# simplices|` array of simplices where `K>1` is the number of vertices per
     * simplex
     */
    PBAT_API void
    Construct(Eigen::Ref<GpuMatrixX const> const& P, Eigen::Ref<GpuIndexMatrixX const> const& S);
    /**
     * @brief Resize the AABB buffer
     * @param dims New embedding dimensionality > 0
     * @param nBoxes New number of boxes
     */
    PBAT_API void Resize(GpuIndex dims, GpuIndex nBoxes);
    /**
     * @brief Get the number of boxes
     * @return Number of boxes
     */
    PBAT_API GpuIndex Size() const;
    /**
     * @brief Get the embedding dimensionality
     * @return Embedding dimensionality
     */
    [[maybe_unused]] GpuIndex Dimensions() const { return mDims; }
    /**
     * @brief Handle to the implementation
     * @return Handle to the implementation
     */
    [[maybe_unused]] void* Impl() { return mImpl; }
    /**
     * @brief Handle to the implementation
     * @return Handle to the implementation
     */
    [[maybe_unused]] void* Impl() const { return mImpl; }
    /**
     * @brief Fetch the lower bounds from GPU and return as CPU matrix
     * @return `dims x |# boxes|` array of lower bounds
     */
    PBAT_API GpuMatrixX Lower() const;
    /**
     * @brief Fetch the upper bounds from GPU and return as CPU matrix
     * @return `dims x |# boxes|` array of upper bounds
     */
    PBAT_API GpuMatrixX Upper() const;
    /**
     * @brief Destroy the Aabb 3 D object
     */
    PBAT_API ~Aabb();

  private:
    void Deallocate();

    GpuIndex mDims;
    void* mImpl;
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_AABB_H
