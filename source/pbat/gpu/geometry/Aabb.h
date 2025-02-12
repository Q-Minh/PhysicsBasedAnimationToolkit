#ifndef PBAT_GPU_GEOMETRY_AABB_H
#define PBAT_GPU_GEOMETRY_AABB_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace geometry {

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

    PBAT_API Aabb(Aabb&& other) noexcept;
    PBAT_API Aabb& operator=(Aabb&& other) noexcept;
    /**
     * @brief
     *
     * @param L
     * @param U
     */
    PBAT_API void
    Construct(Eigen::Ref<GpuMatrixX const> const& L, Eigen::Ref<GpuMatrixX const> const& U);
    /**
     * @brief
     *
     * @param P 3x|#pts| array of points
     * @param S Kx|#simplices| array of simplices where K>1 is the number of vertices per simplex
     */
    PBAT_API void
    Construct(Eigen::Ref<GpuMatrixX const> const& P, Eigen::Ref<GpuIndexMatrixX const> const& S);
    /**
     * @brief
     *
     * @param dims
     * @param nBoxes
     */
    PBAT_API void Resize(GpuIndex dims, GpuIndex nBoxes);
    /**
     * @brief
     *
     * @return 
     */
    PBAT_API GpuIndex Size() const;
    /**
     * @brief
     *
     * @return 
     */
    [[maybe_unused]] GpuIndex Dimensions() const { return mDims; }
    /**
     * @brief
     *
     * @return 
     */
    [[maybe_unused]] void* Impl() { return mImpl; }
    /**
     * @brief
     *
     * @return 
     */
    [[maybe_unused]] void* Impl() const { return mImpl; }
    /**
     * @brief
     *
     * @return 
     */
    PBAT_API GpuMatrixX Lower() const;
    /**
     * @brief
     *
     * @return 
     */
    PBAT_API GpuMatrixX Upper() const;
    /**
     * @brief Destroy the Aabb 3 D object
     *
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
