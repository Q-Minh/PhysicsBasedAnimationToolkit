#ifndef PBAT_GPU_IMPL_VBD_BROYDENINTEGRATOR_CUH
#define PBAT_GPU_IMPL_VBD_BROYDENINTEGRATOR_CUH

#include "Integrator.cuh"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/math/Blas.cuh"
#include "pbat/gpu/impl/math/Matrix.cuh"

namespace pbat::gpu::impl::vbd {

/**
 * @brief BroydenIntegrator accelerated VBD integrator
 */
class BroydenIntegrator : public Integrator
{
  public:
    using BaseType = Integrator; ///< Base class type
    /**
     * @brief Construct a new BroydenIntegrator object
     *
     * @param data VBD simulation scenario
     */
    BroydenIntegrator(Data const& data);
    /**
     * @brief Solve the optimization problem using the Broyden accelerated VBD method
     * @param bdf Device BDF minimization problem
     * @param iterations Number of optimization iterations
     */
    virtual void Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations) override;
    /**
     * @brief Destroy the BroydenIntegrator object
     */
    virtual ~BroydenIntegrator() = default;

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

    math::Matrix<GpuScalar> mQR;  ///< `3|# verts| x m` storage for QR factors
    math::Vector<GpuScalar> mTau; ///< `m` storage for Householder reflector taus
    math::Blas mBlas;             ///< BLAS API for matrix operations
};

} // namespace pbat::gpu::impl::vbd

#endif // PBAT_GPU_IMPL_VBD_BROYDENINTEGRATOR_CUH
