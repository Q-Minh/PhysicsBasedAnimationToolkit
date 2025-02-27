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
     * @brief Rotates \f$ x^k, x^{k-1}, x^{k-2} and f(x^k), f(x^{k-1}), f(x^{k-2}) and t_k, t_{k-1},
     * t_{k-2} \f$
     *
     * Sets \f$ x^{k-1} \leftarrow x^k \f$ and \f$ x^{k-2} \leftarrow x^{k-1} \f$ and
     * \f$ f(x^{k-1}) \leftarrow f(x^k) \f$ and \f$ f(x^{k-2}) \leftarrow f(x^{k-1}) \f$ and
     * \f$ t_{k-1} \leftarrow t_k \f$ and \f$ t_{k-2} \leftarrow t_{k-1} \f$.
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
     * There can be 2 degeneracies:
     * 1. If \f$ a_f = 0, b_f \neq 0 \f$, the function is linear. In this case, we return
     * \f$ t^* = \begin{cases} -\inf & b_f > 0 \\ +\inf & \text{otherwise} \end{cases} \f$
     * 2. If \f$ a_f = b_f = 0 \f$, the function is constant. In this case, we return one of the
     * solutions \f$ t^* = 0 \f$.
     *
     * @return The model function's minimizer
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
     * t_{k-2}^2 & t_{k-2} & 1 \\
     * t_{k-1}^2 & t_{k-1} & 1 \\
     * t_k^2 & t_k & 1
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
     * Because the lead matrix is constant, we can compute its inverse and solve for \f$
     * \mathbf{a_Q}, i.e.
     *
     * \f[
     * \mathbf{a_Q} = \mathbf{Q}^{-1} \mathbf{f}
     * \f]
     *
     * For any 3 consecutive function values \f$ f^{k-2}, f^{k-1}, f^k \f$ at corresponding states
     * \f$ x^{k-2}, x^{k-1}, x^k \f$, we can compute the coefficients \f$ \mathbf{a_Q} \f$ of the
     * quadratic energy proxy function \f$ f_{\text{model}}(t) \f$ as
     * \f$ \mathbf{a_Q} = \mathbf{Q}^{-1} \mathbf{f} \f$.
     *
     * @post `aQ` is such that `ModelFunction(t_{k-2}) = fkm2`, `ModelFunction(t_{k-1}) =
     * fkm1` and `ModelFunction(t_k) = fk`.
     * @post `tkm2, tkm1, tk` are translated such that `tk = 0`
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

    using Matrix3 =
        pbat::math::linalg::mini::SMatrix<GpuScalar, 3, 3>;          ///< Short-hand for 3x3 matrix
    using Vector3 = pbat::math::linalg::mini::SVector<GpuScalar, 3>; ///< Short-hand for 3-vector
    using Vector5 = pbat::math::linalg::mini::SVector<Scalar, 5>;    ///< Short-hand for 5-vector

    /**
     * @brief Compute the coefficients of the polynomial constraint
     * \f$ |\mathbf{x}(t) - \mathbf{x}_k|_2^2 - R^2 = 0 \f$
     *
     * The constraint is a quartic polynomial in \f$ t \f$.
     * The constant coefficient \f$ \sigma_0 = \tilde{\sigma_0} - R^2 \f$, hence we store
     * \f$ \tilde{\sigma_0} \f$ in `sigmax[0]`. This makes it flexible to parameterize the
     * polynomial by the radius \f$ R \f$.
     */
    void ComputePolynomialConstraintCoefficients();
    /**
     * @brief Find the nearest future time \f$ t^* \f$ such that the curved path crosses the trust
     * region's boundary.
     *
     * @param R2 Trust region radius squared
     * @return Nearest future time \f$ t^* \f$ s.t. \f$ |\mathbf{x}(t^*) - \mathbf{x}_k|_2^2 = R^2
     * \f$
     * @pre ComputePolynomialConstraintCoefficients() must be called before this function.
     */
    Scalar SolveCurvedTrustRegionConstraint(Scalar R2) const;
    /**
     * @brief Compute the coefficients of the curved path
     *
     * The coefficients \f$ a_{i,d}, b_{i,d}, c_{i,d} \f$ are computed such that
     * \f[
     * x_{i,d}(t) = a_{i,d} (t-t_k)^2 + b_{i,d} (t-t_k) + c_{i,d}
     * \f]
     *
     * The translation \f$ t - t_k \f$ is done so that the Vandermonde matrix is well-conditioned.
     */
    void ComputeCurvedPath();
    /**
     * @brief Take a quadratic step along the curved accelerated path
     * @param t Step size
     */
    void TakeCurvedStep(GpuScalar t);
    /**
     * @brief Rollback the quadratic step along the curved accelerated path
     */
    void RollbackCurvedStep();

  private:
    GpuScalar eta;                     ///< Trust Region energy reduction accuracy threshold
    GpuScalar tau;                     ///< Trust Region radius increase factor
    common::Buffer<GpuScalar, 3> xkm1; ///< `3x|# verts|` \f$ x^{k-1} \f$
    common::Buffer<GpuScalar, 3> xkm2; ///< `3x|# verts|` \f$ x^{k-2} \f$
    GpuScalar fk, fkm1, fkm2;          ///< objective function
                                       ///< values at \f$ x^k, x^{k-1}, x^{k-2} \f$
    GpuScalar tk, tkm1, tkm2; ///< Points along accelerated path at \f$ x^k, x^{k-1}, x^{k-2} \f$

    Matrix3 Q;        ///< Quadratic energy proxy matrix. See ConstructModel().
    Vector3 aQ;       ///< Quadratic energy proxy coefficients
    GpuScalar am, bm; ///< Linear model function coefficients s.t. \f$ m_k(t) = a_m t + b_m \f$
    Vector5 sigmax;   ///< Trust-Region curved path constraint polynomial coefficients
    common::Buffer<GpuScalar, 3> ax; ///< Curved path quadratic coefficients
    common::Buffer<GpuScalar, 3> bx; ///< Curved path linear coefficients
    common::Buffer<GpuScalar, 3> cx; ///< Curved path constant coefficients
    bool bUseCurvedPath;             ///< Whether to use curved path or not
};

} // namespace pbat::gpu::impl::vbd

#endif // PBAT_GPU_IMPL_VBD_TRUSTREGIONINTEGRATOR_CUH
