#ifndef PBAT_GPU_VBD_VBD_IMPL_CUH
#define PBAT_GPU_VBD_VBD_IMPL_CUH

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/geometry/PrimitivesImpl.cuh"
#include "pbat/gpu/vbd/InitializationStrategy.h"

#include <cuda/api/stream.hpp>
#include <vector>

namespace pbat {
namespace gpu {
namespace vbd {

class VbdImpl
{
  public:
    VbdImpl(
        Eigen::Ref<GpuMatrixX const> const& X,
        Eigen::Ref<GpuIndexMatrixX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& F,
        Eigen::Ref<GpuIndexMatrixX const> const& T);
    /**
     * @brief
     * @param dt
     * @param iterations
     * @param substeps
     * @param rho Chebyshev semi-iterative method's estimated spectral radius. If rho >= 1,
     * Chebyshev acceleration is not used.
     */
    void Step(
        GpuScalar dt,
        GpuIndex iterations,
        GpuIndex substeps = GpuIndex{1},
        GpuScalar rho     = GpuScalar{1});
    /**
     * @brief
     * @param X
     */
    void SetPositions(Eigen::Ref<GpuMatrixX const> const& X);
    /**
     * @brief
     * @param v
     */
    void SetVelocities(Eigen::Ref<GpuMatrixX const> const& v);
    /**
     * @brief
     * @param aext
     */
    void SetExternalAcceleration(Eigen::Ref<GpuMatrixX const> const& aext);
    /**
     * @brief
     * @param m
     */
    void SetMass(Eigen::Ref<GpuVectorX const> const& m);
    /**
     * @brief
     * @param wg
     */
    void SetQuadratureWeights(Eigen::Ref<GpuVectorX const> const& wg);
    /**
     * @brief
     * @param GP
     */
    void SetShapeFunctionGradients(Eigen::Ref<GpuMatrixX const> const& GP);
    /**
     * @brief
     * @param l
     */
    void SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l);
    /**
     * @brief
     * @param zero
     */
    void SetNumericalZeroForHessianDeterminant(GpuScalar zero);
    /**
     * @brief
     * @param GVTp
     * @param GVTn
     * @param GVTilocal
     */
    void SetVertexTetrahedronAdjacencyList(
        Eigen::Ref<GpuIndexVectorX const> const& GVTp,
        Eigen::Ref<GpuIndexVectorX const> const& GVTn,
        Eigen::Ref<GpuIndexVectorX const> const& GVTilocal);
    /**
     * @brief
     * @param kD
     */
    void SetRayleighDampingCoefficient(GpuScalar kD);
    /**
     * @brief
     * @param partitions
     */
    void SetVertexPartitions(std::vector<std::vector<GpuIndex>> const& partitions);
    /**
     * @brief
     * @param strategy
     */
    void SetInitializationStrategy(EInitializationStrategy strategy);
    /**
     * @brief
     * @param blockSize
     */
    void SetBlockSize(GpuIndex blockSize);

    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar, 3> const& GetVelocity() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar, 3> const& GetExternalAcceleration() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar> const& GetMass() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar> const& GetShapeFunctionGradients() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar> const& GetLameCoefficients() const;
    /**
     * @brief
     * @return
     */
    std::vector<common::Buffer<GpuIndex>> const& GetPartitions() const;

  public:
    geometry::PointsImpl X;    ///< Current vertex positions
    geometry::SimplicesImpl V; ///< Boundary vertex simplices
    geometry::SimplicesImpl F; ///< Boundary triangle simplices
    geometry::SimplicesImpl T; ///< Tetrahedral mesh elements
  private:
    common::Buffer<GpuScalar, 3> mPositionsAtT;            ///< Previous vertex positions
    common::Buffer<GpuScalar, 3> mKineticEnergyMinimalPositions; ///< Inertial target for vertex positions
    common::Buffer<GpuScalar, 3>
        mChebyshevPositionsM2; ///< x^{k-2} used in Chebyshev semi-iterative method
    common::Buffer<GpuScalar, 3>
        mChebyshevPositionsM1; ///< x^{k-1} used in Chebyshev semi-iterative method
    common::Buffer<GpuScalar, 3> mVelocitiesAtT;        ///< Previous vertex velocities
    common::Buffer<GpuScalar, 3> mVelocities;           ///< Current vertex velocities
    common::Buffer<GpuScalar, 3> mExternalAcceleration; ///< External acceleration
    common::Buffer<GpuScalar> mMass;                    ///< Lumped mass matrix diagonals

    common::Buffer<GpuScalar> mQuadratureWeights; ///< |#elements| array of quadrature weights (i.e.
                                                  ///< tetrahedron volumes for order 1)
    common::Buffer<GpuScalar> mShapeFunctionGradients; ///< 4x3x|#elements| shape function gradients
    common::Buffer<GpuScalar> mLameCoefficients; ///< 2x|#elements| 1st and 2nd Lame parameters
    GpuScalar mDetHZero;                         ///< Numerical zero for hessian determinant check

    common::Buffer<GpuIndex>
        mVertexTetrahedronPrefix; ///< Vertex-tetrahedron adjacency list's prefix sum
    common::Buffer<GpuIndex>
        mVertexTetrahedronNeighbours; ///< Vertex-tetrahedron adjacency list's neighbour list
    common::Buffer<GpuIndex> mVertexTetrahedronLocalVertexIndices; ///< Vertex-tetrahedron adjacency
                                                                   ///< list's ilocal property

    GpuScalar mRayleighDamping;               ///< Rayleigh damping coefficient
    GpuScalar mCollisionPenalty;              ///< Collision penalty coefficient
    GpuIndex mMaxCollidingTrianglesPerVertex; ///< Memory capacity for storing vertex triangle
                                              ///< collision constraints

    common::Buffer<GpuIndex> mCollidingTriangles; ///< |#vertices|x|mMaxCollidingTrianglesPerVertex|
                                                  ///< array of colliding triangles
    common::Buffer<GpuIndex> mCollidingTriangleCount; ///< |#vertices| array of the number of
                                                      ///< colliding triangles for each vertex.

    std::vector<common::Buffer<GpuIndex>> mPartitions; ///< Constraint partitions

    EInitializationStrategy
        mInitializationStrategy;  ///< Strategy to use to determine the initial BCD iterate
    GpuIndex mGpuThreadBlockSize; ///< Number of threads per CUDA thread block
    cuda::stream_t mStream;       ///< Cuda stream on which this VBD instance will run
};

} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_VBD_IMPL_CUH