#ifndef PBAT_GPU_XPBD_XPBD_H
#define PBAT_GPU_XPBD_XPBD_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"

#include <array>
#include <cstddef>
#include <vector>

namespace pbat {
namespace gpu {
namespace xpbd {

class XpbdImpl;

class Xpbd
{
  public:
    enum class EConstraint : int { StableNeoHookean = 0, Collision };

    // Constructors
    /**
     * @brief
     * @param V
     * @param T
     */
    PBAT_API Xpbd(
        Eigen::Ref<GpuMatrixX const> const& X,
        Eigen::Ref<GpuIndexMatrixX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& F,
        Eigen::Ref<GpuIndexMatrixX const> const& T,
        Eigen::Ref<GpuIndexVectorX const> const& BV,
        Eigen::Ref<GpuIndexVectorX const> const& BF,
        std::size_t nMaxVertexTetrahedronOverlaps,
        std::size_t nMaxVertexTriangleContacts);
    Xpbd(Xpbd const&)            = delete;
    Xpbd& operator=(Xpbd const&) = delete;
    PBAT_API Xpbd(Xpbd&&) noexcept;
    PBAT_API Xpbd& operator=(Xpbd&&) noexcept;
    /**
     * @brief
     */
    PBAT_API void PrepareConstraints();
    /**
     * @brief
     * @param dt
     * @param iterations
     * @params substeps
     */
    PBAT_API void Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps);
    /**
     * @brief
     * @return
     */
    PBAT_API GpuMatrixX Positions() const;
    /**
     * @brief
     * @return
     */
    PBAT_API std::size_t NumberOfParticles() const;
    /**
     * @brief
     * @return
     */
    PBAT_API std::size_t NumberOfConstraints() const;
    /**
     * @brief
     * @param X
     */
    PBAT_API void SetPositions(Eigen::Ref<GpuMatrixX const> const& X);
    /**
     * @brief
     * @param v
     */
    PBAT_API void SetVelocities(Eigen::Ref<GpuMatrixX const> const& v);
    /**
     * @brief
     * @param f
     */
    PBAT_API void SetExternalForces(Eigen::Ref<GpuMatrixX const> const& f);
    /**
     * @brief
     * @param minv
     */
    PBAT_API void SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv);
    /**
     * @brief
     * @param l
     */
    PBAT_API void SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l);
    /**
     * @brief
     * @param alpha
     * @param eConstraint
     */
    PBAT_API void SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint);
    /**
     * @brief
     * @param partitions
     */
    PBAT_API void SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions);
    /**
     * @brief
     * @param muS
     * @param muK
     */
    PBAT_API void SetFrictionCoefficients(GpuScalar muS, GpuScalar muK);
    /**
     * @brief
     * @param min
     * @param max
     */
    PBAT_API void SetSceneBoundingBox(
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max);
    /**
     * @brief
     * @return
     */
    PBAT_API GpuMatrixX GetVelocity() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuMatrixX GetExternalForce() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuVectorX GetMassInverse() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuMatrixX GetLameCoefficients() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuMatrixX GetShapeMatrixInverse() const;
    /**
     * @brief
     * @return
     */
    PBAT_API GpuMatrixX GetRestStableGamma() const;
    /**
     * @brief
     * @param eConstraint
     * @return
     */
    PBAT_API GpuMatrixX GetLagrangeMultiplier(EConstraint eConstraint) const;
    /**
     * @brief
     * @param eConstraint
     * @return
     */
    PBAT_API GpuMatrixX GetCompliance(EConstraint eConstraint) const;
    /**
     * @brief
     * @return
     */
    PBAT_API std::vector<std::vector<GpuIndex>> GetPartitions() const;
    /**
     * @brief Get the vertex-tetrahedron collision candidates list
     *
     * @return GpuIndexMatrixX 2x|#collision candidates|
     */
    PBAT_API GpuIndexMatrixX GetVertexTetrahedronCollisionCandidates() const;
    /**
     * @brief 
     * @return 
     */
    PBAT_API GpuIndexMatrixX GetVertexTriangleContactPairs() const;
    /**
     * @brief
     */
    PBAT_API ~Xpbd();

  private:
    XpbdImpl* mImpl;
};

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_XPBD_XPBD_H