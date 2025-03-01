/**
 * @file Integrator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief VBD integrator public API
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_VBD_INTEGRATOR_H
#define PBAT_GPU_VBD_INTEGRATOR_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Enums.h"

#include <string_view>

namespace pbat::gpu::impl::vbd {
class Integrator;
} // namespace pbat::gpu::impl::vbd

namespace pbat {
namespace gpu {
namespace vbd {

/**
 * @brief The VBD public API wrapper around the VBD integrator implementation.
 * \cite anka2024vbd
 */
class Integrator
{
  public:
    using EInitializationStrategy =
        pbat::sim::vbd::EInitializationStrategy; ///< Initialization strategy
    using Data = pbat::sim::vbd::Data;           ///< VBD data

    /**
     * @brief Construct a new Integrator object
     *
     * @param data VBD data
     */
    PBAT_API
    Integrator(Data const& data);

    Integrator(Integrator const&)            = delete;
    Integrator& operator=(Integrator const&) = delete;
    /**
     * @brief Move constructor
     * @param other Other object
     */
    PBAT_API Integrator(Integrator&& other) noexcept;
    /**
     * @brief Move assignment operator
     * @param other Other object
     * @return Reference to this object
     */
    PBAT_API Integrator& operator=(Integrator&& other) noexcept;
    /**
     * @brief Destructor
     */
    PBAT_API ~Integrator();

    /**
     * @brief Execute one simulation step
     * @param dt Time step
     * @param iterations Number of optimization iterations per substep
     * @param substeps Number of substeps
     */
    PBAT_API void Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps = GpuIndex{1});
    /**
     * @brief Execute one simulation step and trace the result
     *
     * * Saves matrix market files which follow the pattern
     * `{variable}.t.{timestep}.s.{substep}[.k.{iteration}].mtx`
     *
     * @param dt Time step
     * @param iterations Number of optimization iterations per substep
     * @param substeps Number of substeps
     * @param t Current time step
     * @param dir Directory to save the matrix market files
     */
    PBAT_API void TracedStep(
        GpuScalar dt,
        GpuIndex iterations,
        GpuIndex substeps,
        GpuIndex t,
        std::string_view dir = ".");
    /**
     * @brief Set the vertex positions
     * @param X `3x|# vertices|` array of vertex positions
     */
    PBAT_API void SetPositions(Eigen::Ref<GpuMatrixX const> const& X);
    /**
     * @brief Set the vertex velocities
     * @param v `3x|# vertices|` array of vertex velocities
     */
    PBAT_API void SetVelocities(Eigen::Ref<GpuMatrixX const> const& v);
    /**
     * @brief Set the external accelerations
     * @param aext `3x|# vertices|` array of external accelerations
     */
    PBAT_API void SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext);
    /**
     * @brief Set threshold for zero determinant check
     * @param zero Numerical zero used in Hessian determinant check for approximate singularity
     * detection
     */
    PBAT_API void SetNumericalZeroForHessianDeterminant(GpuScalar zero);
    /**
     * @brief Set the damping coefficient for Rayleigh damping
     * @param kD Damping coefficient
     */
    PBAT_API void SetRayleighDampingCoefficient(GpuScalar kD);
    /**
     * @brief Sets the initialization strategy to kick-start the time step minimization
     * @param strategy Initialization strategy
     */
    PBAT_API void SetInitializationStrategy(EInitializationStrategy strategy);
    /**
     * @brief Sets the GPU thread block size, for the BDF1 minimization
     * @param blockSize # threads per block, should be a multiple of 32
     */
    PBAT_API void SetBlockSize(GpuIndex blockSize);
    /**
     * @brief Sets the scene bounding box
     * @param min Minimum corner of the bounding box
     * @param max Maximum corner of the bounding box
     */
    PBAT_API void SetSceneBoundingBox(
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max);
    /**
     * @brief Get the vertex positions
     * @return `|# dims|x|# vertices|` array of vertex positions
     */
    PBAT_API GpuMatrixX GetPositions() const;
    /**
     * @brief Get the vertex velocities
     * @return `|# dims|x|# vertices|` array of vertex velocities
     */
    PBAT_API GpuMatrixX GetVelocities() const;

  private:
    impl::vbd::Integrator* mImpl;
};

} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_INTEGRATOR_H
