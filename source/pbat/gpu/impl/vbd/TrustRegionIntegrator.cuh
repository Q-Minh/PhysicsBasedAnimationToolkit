/**
 * @file TrustRegionIntegrator.cuh
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Trust-Region accelerated VBD integrator
 * @date 2025-02-18
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_IMPL_VBD_TRUSTREGIONINTEGRATOR_CUH
#define PBAT_GPU_IMPL_VBD_TRUSTREGIONINTEGRATOR_CUH

#include "Integrator.cuh"
#include "pbat/math/linalg/mini/Matrix.h"

namespace pbat::gpu::impl::vbd {

/**
 * @brief Trust-Region accelerated VBD integrator
 */
class TrustRegionIntegrator : public Integrator
{
  public:
    /**
     * @brief Construct a new Trust Region Integrator object
     * @param data VBD simulation scenario
     */
    TrustRegionIntegrator(Data const& data);
    /**
     * @brief Solve the optimization problem using the Trust-Region accelerated VBD method
     * @param bdf Device BDF minimization problem
     * @param iterations Number of optimization iterations
     */
    virtual void Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations) override;
    /**
     * @brief Solve the optimization problem using the Trust-Region accelerated VBD method with a
     * linear accelerated path
     * @param bdf Device BDF minimization problem
     * @param iterations Number of optimization iterations
     */
    void
    SolveWithLinearAcceleratedPath(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations);
    /**
     * @brief Solve the optimization problem using the Trust-Region accelerated VBD method with a
     * curved accelerated path
     * @param bdf Device BDF minimization problem
     * @param iterations Number of optimization iterations
     */
    void
    SolveWithCurvedAccelerationPath(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations);
    /**
     * @brief Compute the objective function value at \f$ x^k \f$, i.e.
     *
     * \f[
     * \sum_i \frac{1}{2} m_i |\mathbf{x}_i -
     * \tilde{\mathbf{x}_i}|_2^2 + h^2 \sum_{e} w_e \Psi_e +
     * \sum_{c \in (i,f)}
     * \left[ \frac{1}{2} \mu_C d^2 + \mu_F \lambda_N f_0(|\mathbf{u}|) \right]
     * \f]
     *
     * @param dt Time step
     * @param dt2 Time step squared
     */
    GpuScalar ObjectiveFunction(GpuScalar dt, GpuScalar dt2);
    /**
     * @brief Destroy the Trust Region Integrator object
     */
    virtual ~TrustRegionIntegrator() = default;

    /**
     * Helper functions for Trust-Region accelerated VBD method, made public due to limitations on
     * CUDA's extended device lambdas.
     */

    /**
     * @brief Update \f$ x^k, x^{k-1}, x^{k-2}, f(x^k), f(x^{k-1}), f(x^{k-2}) \f$
     */
    void UpdateIterates();
    /**
     * @brief Compute the Trust-Region proxy objective function
     * @param t Point on accelerated path
     * @return Proxy objective function value
     */
    GpuScalar ProxyObjectiveFunction(GpuScalar t) const;
    /**
     * @brief Update the Trust-Region model function (i.e. the quadratic energy proxy along
     * accelerated path)
     */
    void ConstructProxy();
    /**
     * @brief Compute the squared step size \f$ |x^k - x^{k-1}|_2^2 \f$
     * @return \f$ |x^k - x^{k-1}|_2^2 \f$
     */
    GpuScalar SquaredStepSize() const;
    /**
     * @brief Take a linear step along the accelerated path
     * @param t Step size
     * @post `xtr = xkm1 + t * (x - xkm1)`
     */
    void TakeLinearStep(GpuScalar t);
    /**
     * @brief Rollback the linear step along the accelerated path
     * @param t Step size
     * @post `x = xkm1 + (xtr - xkm1) / t`
     */
    void RollbackLinearStep(GpuScalar t);

  private:
    GpuScalar eta;                     ///< Trust Region energy reduction accuracy threshold
    GpuScalar tau;                     ///< Trust Region radius increase factor
    common::Buffer<GpuScalar, 3> xkm1; ///< `3x|# verts|` \f$ x^{k-1} \f$
    common::Buffer<GpuScalar, 3> xkm2; ///< `3x|# verts|` \f$ x^{k-2} \f$
    GpuScalar fk, fkm1, fkm2;          ///< objective function
                                       ///< values at \f$ x^k, x^{k-1}, x^{k-2} \f$

    pbat::math::linalg::mini::SMatrix<GpuScalar, 3, 3>
        Qinv; ///< Inverse of the quadratic energy proxy matrix
    pbat::math::linalg::mini::SVector<GpuScalar, 3> aQ; ///< Quadratic energy proxy coefficients
    bool bUseCurvedPath;                                ///< Whether to use curved path or not
};

} // namespace pbat::gpu::impl::vbd

#endif // PBAT_GPU_IMPL_VBD_TRUSTREGIONINTEGRATOR_CUH
