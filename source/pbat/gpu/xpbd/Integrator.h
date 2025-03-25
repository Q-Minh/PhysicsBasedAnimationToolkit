/**
 * @file Integrator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains the public API of the GPU XPBD integrator.
 * @date 2025-03-25
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_XPBD_INTEGRATOR_H
#define PBAT_GPU_XPBD_INTEGRATOR_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/sim/xpbd/Data.h"
#include "pbat/sim/xpbd/Enums.h"

#include <array>
#include <vector>

namespace pbat::gpu::impl::xpbd {
class Integrator;
} // namespace pbat::gpu::impl::xpbd

namespace pbat {
namespace gpu {
namespace xpbd {

/**
 * @brief Public API of the GPU XPBD integrator.
 */
class Integrator
{
  public:
    using Data = pbat::sim::xpbd::Data; ///< Input XPBD data constructor parameter object short name
    using EConstraint = pbat::sim::xpbd::EConstraint; ///< Constraint type enumeration short name

    // Constructors
    /**
     * @brief Construct a new Integrator object from the input XPBD data
     * @param data Input XPBD data
     */
    PBAT_API Integrator(Data const& data);
    Integrator(Integrator const&)            = delete;
    Integrator& operator=(Integrator const&) = delete;
    /**
     * @brief Move constructor
     * @param other Integrator to move from
     */
    PBAT_API Integrator(Integrator&&) noexcept;
    /**
     * @brief Move assignment operator
     * @param other Integrator to move from
     * @return Reference to this Integrator
     */
    PBAT_API Integrator& operator=(Integrator&&) noexcept;
    /**
     * @brief Step once in time
     * @param dt Time step
     * @param iterations Number of solver loop iterations
     * @params substeps Number of substeps
     */
    PBAT_API void Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps);
    /**
     * @brief Fetch the current positions from the GPU
     * @return `3x|# vertices|` matrix of vertex positions
     */
    PBAT_API GpuMatrixX Positions() const;
    /**
     * @brief Set the GPU vertex positions from the input CPU vertex positions
     * @param X `3x|# vertices|` matrix of vertex positions
     */
    PBAT_API void SetPositions(Eigen::Ref<GpuMatrixX const> const& X);
    /**
     * @brief Set the GPU vertex velocities from the input CPU vertex velocities
     * @param v `3x|# vertices|` matrix of vertex velocities
     */
    PBAT_API void SetVelocities(Eigen::Ref<GpuMatrixX const> const& v);
    /**
     * @brief Set the GPU external acceleration from the input CPU external acceleration
     * @param aext `3x|# vertices|` matrix of external accelerations
     */
    PBAT_API void SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext);
    /**
     * @brief Set the GPU mass inverse from the input CPU mass inverse
     * @param minv `|# vertices|` vector of mass inverses
     */
    PBAT_API void SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv);
    /**
     * @brief Set the GPU Lame coefficients from the input CPU Lame coefficients
     * @param l `2x|# elements|` vector of Lame coefficients
     */
    PBAT_API void SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l);
    /**
     * @brief Set the GPU constraint compliances from the input CPU constraint compliances
     * @param alpha `|# constraints|` matrix of constraint compliances of type eConstraint
     * @param eConstraint Type of constraint
     */
    PBAT_API void SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint);
    /**
     * @brief Set the friction coefficients to use for contact handling
     * @param muS Static friction coefficient
     * @param muK Kinetic friction coefficient
     */
    PBAT_API void SetFrictionCoefficients(GpuScalar muS, GpuScalar muK);
    /**
     * @brief Set the world bounding box for the current time step
     * @param min World bounding box minimum
     * @param max World bounding box maximum
     */
    PBAT_API void SetSceneBoundingBox(
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max);
    /**
     * @brief Fetch the vertex velocities from the GPU
     * @return `3x|# vertices|` matrix of vertex velocities
     */
    PBAT_API GpuMatrixX GetVelocity() const;
    /**
     * @brief Fetch the external acceleration from the GPU
     * @return `3x|# vertices|` matrix of external accelerations
     */
    PBAT_API GpuMatrixX GetExternalAcceleration() const;
    /**
     * @brief Fetch the mass inverse from the GPU
     * @return `|# vertices|` vector of mass inverses
     */
    PBAT_API GpuVectorX GetMassInverse() const;
    /**
     * @brief Fetch the Lame coefficients from the GPU
     * @return `2x|# elements|` vector of Lame coefficients
     */
    PBAT_API GpuMatrixX GetLameCoefficients() const;
    /**
     * @brief Fetch the shape matrix inverses from the GPU
     * @return `3x3x|# elements|` matrix of shape matrix inverses
     */
    PBAT_API GpuMatrixX GetShapeMatrixInverse() const;
    /**
     * @brief Fetch the Stable Neo-Hookean rest-stable gamma from the GPU \cite smith2018snh
     * @return `|# elements|` vector of rest-stable gamma
     */
    PBAT_API GpuMatrixX GetRestStableGamma() const;
    /**
     * @brief Fetch the Lagrange multipliers from the GPU
     * @param eConstraint Type of constraint
     * @return `|# constraints|` vector of Lagrange multipliers for the given constraint type
     */
    PBAT_API GpuMatrixX GetLagrangeMultiplier(EConstraint eConstraint) const;
    /**
     * @brief Fetch the compliance from the GPU
     * @param eConstraint Type of constraint
     * @return `|# constraints|` vector of compliance for the given constraint type
     */
    PBAT_API GpuMatrixX GetCompliance(EConstraint eConstraint) const;
    /**
     * @brief
     */
    PBAT_API ~Integrator();

  private:
    impl::xpbd::Integrator* mImpl;
};

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_XPBD_INTEGRATOR_H
