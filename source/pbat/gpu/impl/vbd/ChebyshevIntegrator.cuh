/**
 * @file ChebyshevIntegrator.cuh
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Chebyshev semi-iterated method accelerated VBD integrator
 * @date 2025-02-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_IMPL_VBD_CHEBYSHEVINTEGRATOR_CUH
#define PBAT_GPU_IMPL_VBD_CHEBYSHEVINTEGRATOR_CUH

#include "Integrator.cuh"
#include "pbat/gpu/impl/common/Buffer.cuh"

namespace pbat::gpu::impl::vbd {

/**
 * @brief Chebyshev semi-iterated method accelerated VBD integrator
 */
class ChebyshevIntegrator : public Integrator
{
  public:
    /**
     * @brief Construct a new Chebyshev Integrator object
     *
     * @param data VBD simulation scenario
     */
    ChebyshevIntegrator(Data const& data);
    /**
     * @brief Solve the optimization problem using the Chebyshev accelerated VBD method
     * @param bdf Device BDF minimization problem
     * @param iterations Number of optimization iterations
     */
    virtual void Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations) override;
    /**
     * @brief Update \f$ x^k, x^{k-1}, x^{k-2} \f$
     *
     * @param k Current iteration
     * @param omega Chebyshev semi-iterative method's relaxation parameter
     */
    void UpdateIterates(GpuIndex k, GpuScalar omega);
    /**
     * @brief Destroy the Chebyshev Integrator object
     */
    virtual ~ChebyshevIntegrator() = default;

  private:
    GpuScalar rho; ///< Chebyshev semi-iterative method's estimated spectral radius. `rho < 1`.
    common::Buffer<GpuScalar, 3> xkm1; ///< \f$ x^{k-1} \f$
    common::Buffer<GpuScalar, 3> xkm2; ///< \f$ x^{k-2} \f$
};

} // namespace pbat::gpu::impl::vbd

#endif // PBAT_GPU_IMPL_VBD_CHEBYSHEVINTEGRATOR_CUH
