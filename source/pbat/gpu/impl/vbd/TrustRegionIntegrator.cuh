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
     * @brief Compute the objective function value at \f$ x^k \f$
     *
     * \f[
     * f(\mathbf{x}) =
     * \sum_i \frac{1}{2} m_i |\mathbf{x}_i -
     * \tilde{\mathbf{x}_i}|_2^2 + h^2 \sum_{e} w_e \Psi_e +
     * \sum_{c \in (i,f)}
     * \left[ \frac{1}{2} \mu_C d^2 + \mu_F \lambda_N f_0(|\mathbf{u}|) \right]
     * \f]
     * is evaluated using parallel reductions over:
     * 1. vertices
     * 2. elements
     * 3. contacts
     *
     * See \cite anka2024vbd and \cite li2020ipc for details.
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
     * @brief Rotates \f$ x^k, x^{k-1}, x^{k-2} and f(x^k), f(x^{k-1}), f(x^{k-2}) \f$
     *
     * Sets \f$ x^{k-1} \leftarrow x^k \f$ and \f$ x^{k-2} \leftarrow x^{k-1} \f$ and
     * \f$ f(x^{k-1}) \leftarrow f(x^k) \f$ and \f$ f(x^{k-2}) \leftarrow f(x^{k-1}) \f$.
     */
    void UpdateIterates();
    /**
     * @brief Compute the Trust-Region model function \f$ f_{\text{model}}(t) \f$
     *
     * See ConstructModel() for details.
     *
     * @param t Point on accelerated path
     * @return Model function value
     */
    GpuScalar ModelFunction(GpuScalar t) const;
    /**
     * @brief Compute the model function's minimizer \f$ t^* = \text{arg}\min_t m(t) \f$
     *
     * With \f$ m(t) = a_f t^2 + b_f t + c_f \f$, the minimizer is easily found by root-finding
     * the stationary condition \f$ m'(t^*) = 2a_ft+b_f = 0 \f$, which yields 
     * \f[
     * t^* = -\frac{b_f}{2 a_f}
     * \f]
     * 
     * @return \f$ t^* \f$
     */
    GpuScalar ModelOptimalStep() const;
    /**
     * @brief Update the Trust-Region model function (i.e. the quadratic energy proxy along
     * accelerated path)
     *
     * Compute \f$ \mathbf{a_Q}=\begin{bmatrix} a_f & b_f & b_f \end{bmatrix} \f$ s.t.
     * \f[
     * f_{\text{model}}(t) = \mathbf{a_Q}^T \mathbf{P}_2(t) = a_f t^2 + b_f t + c_f
     * \f]
     * and \f$ f_{\text{model}}(-1) = f^{k-2} \f$, \f$ f_{\text{model}}(0) = f^{k-1} \f$, \f$
     * f_{\text{model}}(1) = f^k \f$ .
     *
     * This leads to the system of equations
     * \f[
     * \begin{bmatrix}
     * 1 & -1 & 1 \\
     * 0 & 0 & 1 \\
     * 1 & 1 & 1
     * \end{bmatrix}
     * \begin{bmatrix} a_f \\ b_f \\ c_f \end{bmatrix} =
     * \begin{bmatrix} f(\mathbf{x}^{k-2}) \\ f(\mathbf{x}^{k-1}) \\ f(\mathbf{x}^k) \end{bmatrix}
     * \f]
     *
     * or equivalently
     * \f[
     * \mathbf{Q} \mathbf{a_Q} = \mathbf{f}
     * \f]
     *
     * Because the lead matrix is constant, we can precompute its inverse \f$ \mathbf{Q}^{-1} \f$,
     * which we store as 
     * 
     * \f[
     * \mathbf{Q}^{-1} = 
     * \begin{bmatrix}
     * 0.5 & -1 & 0.5 \\
     * -0.5 & 0 & 0.5 \\
     * 0 & 1 & 0
     * \end{bmatrix}
     * \f]
     *
     * For any 3 consecutive function values \f$ f^{k-2}, f^{k-1}, f^k \f$ at corresponding states
     * \f$ x^{k-2}, x^{k-1}, x^k \f$, we can compute the coefficients \f$ \mathbf{a_Q} \f$ of the
     * quadratic energy proxy function \f$ f_{\text{model}}(t) \f$ as
     * \f$ \mathbf{a_Q} = \mathbf{Q}^{-1} \mathbf{f} \f$.
     *
     * @post `aQ` is such that `ModelFunction(-1) = fkm2`, `ModelFunction(0) =
     * fkm1` and `ModelFunction(1) = fk`.
     */
    void ConstructModel();
    /**
     * @brief Compute the squared step size \f$ |x^k - x^{k-1}|_2^2 \f$
     *
     * \f[
     * |x^k - x^{k-1}|_2^2 = \sum_i |\mathbf{x}_i^k - \mathbf{x}_i^{k-1}|_2^2
     * \f]
     * which we compute via parallel reduction over vertices.
     *
     * @return \f$ |x^k - x^{k-1}|_2^2 \f$
     */
    GpuScalar SquaredStepSize() const;
    /**
     * @brief Take a linear step along the accelerated path
     * @param t Step size
     *
     * Computes the linear trust-region accelerated step as
     * \f[
     * x_{\text{TR}} = x^{k-1} + t (x^k - x^{k-1})
     * \f]
     * using corresponding states in `xkm1` and `x`.
     * Then, overwrites `x` with `xtr`.
     *
     * @post `x <- xkm1 + t * (x - xkm1)`
     */
    void TakeLinearStep(GpuScalar t);
    /**
     * @brief Rollback the linear step along the accelerated path
     * @param t Step size
     *
     * Computes the rollback of the linear trust-region accelerated step as
     * \f[
     * x = x^{k-1} + \frac{x_{\text{TR}} - x^{k-1}}{t}
     * \f]
     * using corresponding states in `xkm1` and `x`, where `x` currently stores \f$ x_{\text{TR}}
     * \f$. Then, overwrites `x`.
     *
     * @note See TakeLinearStep() for the forward step.
     *
     * @post `x <- xkm1 + (x - xkm1) / t`
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
        Qinv; ///< Inverse of the quadratic energy proxy matrix. See ConstructModel().
    pbat::math::linalg::mini::SVector<GpuScalar, 3> aQ; ///< Quadratic energy proxy coefficients
    bool bUseCurvedPath;                                ///< Whether to use curved path or not
};

} // namespace pbat::gpu::impl::vbd

#endif // PBAT_GPU_IMPL_VBD_TRUSTREGIONINTEGRATOR_CUH
