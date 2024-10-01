#ifndef PBAT_GPU_VBD_VBD_H
#define PBAT_GPU_VBD_VBD_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/vbd/InitializationStrategy.h"

namespace pbat {
namespace gpu {
namespace vbd {

class VbdImpl;

class Vbd
{
  public:
    PBAT_API Vbd(Eigen::Ref<GpuMatrixX const> const& X,
        Eigen::Ref<GpuIndexMatrixX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& F,
        Eigen::Ref<GpuIndexMatrixX const> const& T);

    Vbd(Vbd const&)            = delete;
    Vbd& operator=(Vbd const&) = delete;
    PBAT_API Vbd(Vbd&& other) noexcept;
    PBAT_API Vbd& operator=(Vbd&& other) noexcept;
    PBAT_API ~Vbd();

    /**
     * @brief
     * @param dt
     * @param iterations
     * @param substeps
     * @param rho Chebyshev semi-iterative method's estimated spectral radius. If rho >= 1,
     * Chebyshev acceleration is not used.
     */
    PBAT_API void Step(
        GpuScalar dt,
        GpuIndex iterations,
        GpuIndex substeps = GpuIndex{1},
        GpuScalar rho     = GpuScalar{1});
    /**
     * @brief
     * @param X 3x|#vertices| array of vertex positions
     */
    PBAT_API void SetPositions(Eigen::Ref<GpuMatrixX const> const& X);
    /**
     * @brief
     * @param v 3x|#vertices| array of vertex velocities
     */
    PBAT_API void SetVelocities(Eigen::Ref<GpuMatrixX const> const& v);
    /**
     * @brief
     * @param aext 3x|#vertices| array of external accelerations
     */
    PBAT_API void SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext);
    /**
     * @brief
     * @param m |#vertices| array of lumped masses
     */
    PBAT_API void SetMass(Eigen::Ref<GpuVectorX const> const& m);
    /**
     * @brief
     * @param wg |#elements| array of quadrature weights
     */
    PBAT_API void SetQuadratureWeights(Eigen::Ref<GpuVectorX const> const& wg);
    /**
     * @brief
     * @param GP |4x3|x|#elements| array of shape function gradients
     */
    PBAT_API void SetShapeFunctionGradients(Eigen::Ref<GpuMatrixX const> const& GP);
    /**
     * @brief
     * @param l 2x|#elements| array of lame coefficients l, where l[0,:] = mu and l[1,:] = lambda
     */
    PBAT_API void SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l);
    /**
     * @brief
     * @param zero Numerical zero used in Hessian determinant check for approximate singularity
     * detection
     */
    PBAT_API void SetNumericalZeroForHessianDeterminant(GpuScalar zero);
    /**
     * @brief Sets the adjacency graph of vertices to incident tetrahedra in compressed column
     * format
     * @param GVTp
     * @param GVTn
     * @param GVTilocal
     */
    PBAT_API void SetVertexTetrahedronAdjacencyList(
        Eigen::Ref<GpuIndexVectorX const> const& GVTp,
        Eigen::Ref<GpuIndexVectorX const> const& GVTn,
        Eigen::Ref<GpuIndexVectorX const> const& GVTilocal);
    /**
     * @brief
     * @param kD
     */
    PBAT_API void SetRayleighDampingCoefficient(GpuScalar kD);
    /**
     * @brief Sets the groups/partitions of vertices that can be minimized independently, i.e. in
     * parallel.
     * @param partitions
     */
    PBAT_API void SetVertexPartitions(std::vector<std::vector<GpuIndex>> const& partitions);
    /**
     * @brief Sets the initialization strategy to kick-start the time step minimization
     * @param strategy
     */
    PBAT_API void SetInitializationStrategy(EInitializationStrategy strategy);
    /**
     * @brief Sets the GPU thread block size, for the BDF1 minimization
     * @param blockSize #threads per block, should be a multiple of 32
     */
    PBAT_API void SetBlockSize(GpuIndex blockSize);
    /**
     * @brief
     * @return |#dims|x|#vertices| array of vertex positions
     */
    PBAT_API GpuMatrixX GetPositions() const;
    /**
     * @brief
     * @return |#dims|x|#vertices| array of vertex velocities
     */
    PBAT_API GpuMatrixX GetVelocities() const;

  private:
    VbdImpl* mImpl;
};

} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_VBD_H