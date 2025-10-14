/**
 * @file AndersonIntegrator.cuh
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Anderson acceleration VBD integrator
 * @date 2025-04-15
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_IMPL_VBD_ANDERSONINTEGRATOR_CUH
#define PBAT_GPU_IMPL_VBD_ANDERSONINTEGRATOR_CUH

#include "Integrator.cuh"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/math/Blas.cuh"
#include "pbat/gpu/impl/math/LinearSolver.cuh"
#include "pbat/gpu/impl/math/Matrix.cuh"

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
     * @brief Destroy the AndersonIntegrator object
     */
    virtual ~AndersonIntegrator() = default;
    /**
     * @brief Update the Anderson window with the current residual and iterate
     * @param k Current iteration index
     */
    void UpdateAndersonWindow(GpuIndex k);
    /**
     * @brief Take an Anderson accelerated step
     * @param k Current iteration index
     */
    void TakeAndersonAcceleratedStep(GpuIndex k);
    /**
     * @brief Add regularization to the diagonal of the QR factorization
     * @param mk Current window size
     */
    void RegularizeRFactor(GpuIndex mk);

  private:
    math::Vector<GpuScalar> Fk;   ///< `3|# verts| x 1` vector of current residual
    math::Vector<GpuScalar> Fkm1; ///< `3|# verts| x 1` vector of past residual
    math::Vector<GpuScalar> Gk;   ///< `3|# verts|` vector of current iterate
    math::Vector<GpuScalar> Gkm1; ///< `3|# verts| x 1` vector of past iterate
    math::Vector<GpuScalar> xkm1; ///< `3|# verts|` past iterate vector
    math::Matrix<GpuScalar>
        mDFK; ///< `3|# verts| x m` matrix of past residuals window used in Anderson acceleration
    math::Matrix<GpuScalar>
        mDGK; ///< `3|# verts| x m` matrix of past iterates window used in Anderson acceleration

    math::Matrix<GpuScalar> mQR;      ///< `3|# verts| x m` storage for QR factors
    math::Vector<GpuScalar> mTau;     ///< `m` storage for Householder reflector taus
    math::Blas mBlas;                 ///< BLAS API for matrix operations
    math::LinearSolver mLinearSolver; ///< Linear solver API for QR solve
    common::Buffer<GpuScalar> mLinearSolverWorkspace; ///< Linear solver workspace buffer
};

} // namespace pbat::gpu::impl::vbd

#endif // PBAT_GPU_IMPL_VBD_ANDERSONINTEGRATOR_CUH
