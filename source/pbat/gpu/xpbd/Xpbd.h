#ifndef PBAT_GPU_XPBD_XPBD_H
#define PBAT_GPU_XPBD_XPBD_H

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
    Xpbd(
        Eigen::Ref<GpuMatrixX const> const& X,
        Eigen::Ref<GpuIndexMatrixX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& F,
        Eigen::Ref<GpuIndexMatrixX const> const& T,
        std::size_t nMaxVertexTriangleOverlaps,
        GpuScalar kMaxCollisionPenetration = GpuScalar{1.});
    Xpbd(Xpbd const&)            = delete;
    Xpbd& operator=(Xpbd const&) = delete;
    Xpbd(Xpbd&&) noexcept;
    Xpbd& operator=(Xpbd&&) noexcept;
    /**
     * @brief
     */
    void PrepareConstraints();
    /**
     * @brief
     * @param dt
     * @param iterations
     * @params substeps
     */
    void Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps);
    /**
     * @brief
     * @return
     */
    GpuMatrixX Positions() const;
    /**
     * @brief
     * @return
     */
    std::size_t NumberOfParticles() const;
    /**
     * @brief
     * @return
     */
    std::size_t NumberOfConstraints() const;
    /**
     * @brief
     * @param X
     */
    void SetPositions(Eigen::Ref<GpuMatrixX const> const& X);
    /**
     * @brief
     * @param v
     */
    void SetVelocities(Eigen::Ref<GpuMatrixX const> const& v);
    /**
     * @brief
     * @param f
     */
    void SetExternalForces(Eigen::Ref<GpuMatrixX const> const& f);
    /**
     * @brief
     * @param minv
     */
    void SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv);
    /**
     * @brief
     * @param l
     */
    void SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l);
    /**
     * @brief
     * @param alpha
     * @param eConstraint
     */
    void SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint);
    /**
     * @brief
     * @param partitions
     */
    void SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions);
    /**
     * @brief
     * @param kMaxCollisionPenetration
     */
    void SetMaxCollisionPenetration(GpuScalar kMaxCollisionPenetration);
    /**
     * @brief
     * @param muS
     * @param muK
     */
    void SetFrictionCoefficients(GpuScalar muS, GpuScalar muK);
    /**
     * @brief
     * @return
     */
    GpuMatrixX GetVelocity() const;
    /**
     * @brief
     * @return
     */
    GpuMatrixX GetExternalForce() const;
    /**
     * @brief
     * @return
     */
    GpuVectorX GetMassInverse() const;
    /**
     * @brief
     * @return
     */
    GpuMatrixX GetLameCoefficients() const;
    /**
     * @brief
     * @return
     */
    GpuMatrixX GetShapeMatrixInverse() const;
    /**
     * @brief
     * @return
     */
    GpuMatrixX GetRestStableGamma() const;
    /**
     * @brief
     * @param eConstraint
     * @return
     */
    GpuMatrixX GetLagrangeMultiplier(EConstraint eConstraint) const;
    /**
     * @brief
     * @param eConstraint
     * @return
     */
    GpuMatrixX GetCompliance(EConstraint eConstraint) const;
    /**
     * @brief
     * @return
     */
    std::vector<std::vector<GpuIndex>> GetPartitions() const;
    /**
     * @brief Get the Vertex Triangle Overlaps list
     *
     * @return GpuIndexMatrixX 2x|#overlap candidates|
     */
    GpuIndexMatrixX GetVertexTriangleOverlaps() const;
    /**
     * @brief
     */
    ~Xpbd();

  private:
    XpbdImpl* mImpl;
};

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_XPBD_XPBD_H