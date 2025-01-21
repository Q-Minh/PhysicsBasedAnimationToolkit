#ifndef PBAT_GPU_XPBD_INTEGRATOR_H
#define PBAT_GPU_XPBD_INTEGRATOR_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/sim/xpbd/Data.h"
#include "pbat/sim/xpbd/Enums.h"

#include <array>
#include <vector>

namespace pbat {
namespace gpu {
namespace xpbd {
namespace impl {

class Integrator;

} // namespace impl

class Integrator
{
  public:
    using Data        = pbat::sim::xpbd::Data;
    using EConstraint = pbat::sim::xpbd::EConstraint;

    // Constructors
    /**
     * @brief
     * @param V
     * @param T
     */
    PBAT_API Integrator(
        Data const& data,
        std::size_t nMaxVertexTetrahedronOverlaps,
        std::size_t nMaxVertexTriangleContacts);
    Integrator(Integrator const&)            = delete;
    Integrator& operator=(Integrator const&) = delete;
    PBAT_API Integrator(Integrator&&) noexcept;
    PBAT_API Integrator& operator=(Integrator&&) noexcept;
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
     * @param aext
     */
    PBAT_API void SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext);
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
    PBAT_API GpuMatrixX GetExternalAcceleration() const;
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
    PBAT_API ~Integrator();

  private:
    impl::Integrator* mImpl;
};

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_XPBD_INTEGRATOR_H
