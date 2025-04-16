/**
 * @file AndersonIntegrator.cuh
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Anderson acceleration VBD integrator
 * @date 2025-04-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_IMPL_VBD_ANDERSONINTEGRATOR_H
#define PBAT_GPU_IMPL_VBD_ANDERSONINTEGRATOR_H

#include "Integrator.cuh"
#include "pbat/gpu/impl/common/Buffer.cuh"

namespace pbat::gpu::impl::vbd {

/**
 * @brief AndersonIntegrator accelerated VBD integrator
 */
class AndersonIntegrator : public Integrator
{
  public:
    /**
     * @brief Construct a new AndersonIntegrator object
     *
     * @param data VBD simulation scenario
     */
    AndersonIntegrator(Data const& data);
    /**
     * @brief Solve the optimization problem using the Anderson accelerated VBD method
     * @param bdf Device BDF minimization problem
     * @param iterations Number of optimization iterations
     */
    virtual void Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations) override;
    /**
     * @brief Destroy the Chebyshev Integrator object
     */
    virtual ~AndersonIntegrator() = default;

  private:
    common::Buffer<GpuScalar> Fk;   ///< `3|# verts| x 1` vector of current residual
    common::Buffer<GpuScalar> Fkm1; ///< `3|# verts| x 1` vector of past residual
    common::Buffer<GpuScalar> Gkm1; ///< `3|# verts| x 1` vector of past iterate
    common::Buffer<GpuScalar> xkm1; ///< `3|# verts|` past iterate vector
    common::Buffer<GpuScalar> DFK; ///< `3|# verts| x m` matrix of past residuals window used in Anderson acceleration
    common::Buffer<GpuScalar> DGK; ///< `3|# verts| x m` matrix of past iterates window used in Anderson acceleration
    common::Buffer<GpuScalar> alpha; ///< `m` vector of Anderson coefficients
};

} // namespace pbat::gpu::impl::vbd

#endif // PBAT_GPU_IMPL_VBD_ANDERSONINTEGRATOR_H
