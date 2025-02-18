/**
 * @file Integrator.cuh
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief VBD integrator implementation
 * @date 2025-02-14
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_IMPL_VBD_INTEGRATOR_H
#define PBAT_GPU_IMPL_VBD_INTEGRATOR_H

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/contact/VertexTriangleMixedCcdDcd.cuh"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Enums.h"
// clang-format off
/**
 * @warning Kernels.cuh includes cuda-api-wrappers headers, whose cuda::span interferes with 
 * libcu++ (i.e. libcudacxx) cuda::span. To avoid compilation errors, we include Kernels.cuh after all other
 * headers, and we do not use cuda::span when both libcu++ and cuda-api-wrappers headers are present.
 */
#include "Kernels.cuh"
// clang-format on

#include <cuda/api/stream.hpp>
#include <vector>

namespace pbat {
namespace gpu {
namespace impl {
namespace vbd {

/**
 * @brief VBD integrator \cite anka2024vbd
 *
 */
class Integrator
{
  public:
    using EInitializationStrategy =
        pbat::sim::vbd::EInitializationStrategy; ///< Enum type for initialization strategy
    using Data = pbat::sim::vbd::Data;           ///< Data type for VBD

    /**
     * @brief Construct Integrator from data
     *
     * @param data VBD simulation scenario
     */
    Integrator(Data const& data);
    /**
     * @brief Execute one simulation step
     * @param dt Time step
     * @param iterations Number of optimization iterations per substep
     * @param substeps Number of substeps
     * @param rho Chebyshev semi-iterative method's estimated spectral radius. If rho >= 1,
     * Chebyshev acceleration is not used.
     */
    void Step(
        GpuScalar dt,
        GpuIndex iterations,
        GpuIndex substeps = GpuIndex{1},
        GpuScalar rho     = GpuScalar{1});
    /**
     * @brief Set the bounding box over the scene used for spatial partitioning
     * @param min Minimum corner
     * @param max Maximum corner
     */
    void SetSceneBoundingBox(
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max);
    /**
     * @brief Set the number of threads per GPU block for per-vertex elastic energy computation
     * @param blockSize Number of threads per block
     */
    void SetBlockSize(GpuIndex blockSize);

    /**
     * @brief Detect candidate contact pairs for the current time step
     *
     * @param dt Time step
     */
    void InitializeActiveSet(GpuScalar dt);
    /**
     * @brief Compute the inertial target positions of the BDF1 objective
     *
     * @param sdt Time step
     * @param sdt2 Time step squared
     */
    void ComputeInertialTargets(GpuScalar sdt, GpuScalar sdt2);
    /**
     * @brief Compute starting point of BCD minimization
     *
     * @param sdt Time step
     * @param sdt2 Time step squared
     */
    void InitializeBcdSolution(GpuScalar sdt, GpuScalar sdt2);
    /**
     * @brief Computes active contact pairs in the current configuration
     */
    void UpdateActiveSet();
    /**
     * @brief Use VBD to iterate on the BDF minimization problem
     *
     * @param bdf Device BDF minimization problem
     * @param iterations Number of iterations
     * @param rho Chebyshev semi-iterative method's estimated spectral radius. If `rho >= 1`,
     * Chebyshev acceleration is not used.
     */
    void Solve(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations, GpuScalar rho);
    /**
     * @brief Solve the BDF minimization problem using the vanilla VBD method
     *
     * @param bdf The device BDF minimization problem
     * @param iterations Number of iterations
     */
    void SolveWithVanillaVbd(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations);
    /**
     * @brief Use Chebyshev semi-iterative method to accelerate the BDF minimization
     *
     * @param bdf Device BDF minimization problem
     * @param iterations Number of iterations
     * @param rho Chebyshev semi-iterative method's estimated spectral radius
     */
    void SolveWithChebyshevVbd(
        kernels::BackwardEulerMinimization& bdf,
        GpuIndex iterations,
        GpuScalar rho);
    /**
     * @brief Use Trust Region method to accelerate VBD's BDF minimization
     *
     * @param bdf Device BDF minimization problem
     * @param iterations Number of iterations
     */
    void
    SolveWithLinearTrustRegionVbd(kernels::BackwardEulerMinimization& bdf, GpuIndex iterations);
    /**
     * @brief Run a single iteration of the VBD's BDF minimization
     *
     * @param bdf Device BDF minimization problem
     */
    void RunVbdIteration(kernels::BackwardEulerMinimization& bdf);
    /**
     * @brief Update the BDF state (i.e. positions and velocities) after solving the BDF
     * minimization
     *
     * @param sdt Time step
     */
    void UpdateBdfState(GpuScalar sdt);
    /**
     * @brief Compute the per-vertex proxy energies \f$ \frac{1}{2} m_i |\mathbf{x}_i -
     * \tilde{\mathbf{x}_i}|_2^2 + h^2 \sum_{e \in \text{adj}(i)} w_e \Psi_e + \sum_{c \in (i,f)}
     * \left[ \frac{1}{2} \mu_C d^2 + \mu_F \lambda_N f_0(|\mathbf{u}|) \right] \f$
     *
     * @param bdf Device BDF minimization problem
     */
    void ComputeVertexEnergies(kernels::BackwardEulerMinimization& bdf);
    /**
     * @brief Update \f$ x^k, x^{k-1}, x^{k-2} \f$ using Chebyshev acceleration
     *
     * @param k Current iteration
     * @param omega Chebyshev semi-iterative method's relaxation parameter
     */
    void UpdateChebyshevIterates(GpuIndex k, GpuScalar omega);
    /**
     * @brief Update \f$ x^k, x^{k-1}, x^{k-2} \f$
     */
    void UpdateTrustRegionIterates();
    /**
     * @brief Compute the per-element elastic energies
     */
    void ComputeElementElasticEnergies();

  public:
    common::Buffer<GpuScalar, 3> x; ///< Vertex positions
    common::Buffer<GpuIndex, 4> T;  ///< Tetrahedral mesh elements

    Eigen::Vector<GpuScalar, 3> mWorldMin; ///< World AABB min
    Eigen::Vector<GpuScalar, 3> mWorldMax; ///< World AABB max
    GpuIndex mActiveSetUpdateFrequency;    ///< Active set update frequency
    contact::VertexTriangleMixedCcdDcd cd; ///< Collision detector
    common::Buffer<GpuIndex>
        fc; ///< `|# verts * kMaxCollidingTrianglesPerVertex|` array of vertex-triangle contact
            ///< pairs `(i,f)`, s.t. `f = fc[i*kMaxCollidingTrianglesPerVertex+c]` for `0 <= c <
            ///< kMaxCollidingTrianglesPerVertex`. If `f(c) < 0`, there is no contact, and `f(c+j) <
            ///< 0` is also true, for `j > 0`.
    common::Buffer<GpuScalar> XVA; ///< `|x.Size()|` array of vertex areas for contact response
    common::Buffer<GpuScalar> FA;  ///< `|F.Size()|` array of triangle areas for contact response

    common::Buffer<GpuScalar, 3> mPositionsAtT;            ///< Previous vertex positions
    common::Buffer<GpuScalar, 3> mInertialTargetPositions; ///< Inertial target for vertex positions
    common::Buffer<GpuScalar, 3>
        xkm2; ///< \f$ x^{k-2} \f$ used in acceleration schemes (Chebyshev, Trust Region)
    common::Buffer<GpuScalar, 3>
        xkm1; ///< \f$ x^{k-1} \f$ used in acceleration schemes (Chebyshev, Trust Region)
    common::Buffer<GpuScalar> Uetr;   ///< `|# elements|` elastic energies (used for Trust Region
                                      ///< acceleration)
    common::Buffer<GpuScalar, 3> ftr; ///< `3 x |# verts|` per-vertex objective function values
                                      ///< (used for Trust Region acceleration)
    common::Buffer<GpuScalar, 3> xb;  ///< Write buffer for positions which handles data races

    common::Buffer<GpuScalar, 3> mVelocitiesAtT;        ///< Previous vertex velocities
    common::Buffer<GpuScalar, 3> mVelocities;           ///< Current vertex velocities
    common::Buffer<GpuScalar, 3> mExternalAcceleration; ///< External acceleration
    common::Buffer<GpuScalar> mMass;                    ///< Lumped mass matrix diagonals

    common::Buffer<GpuScalar> mQuadratureWeights; ///< `|# elements|` array of quadrature weights
                                                  ///< (i.e. tetrahedron volumes for order 1)
    common::Buffer<GpuScalar>
        mShapeFunctionGradients;                 ///< `4x3x|# elements|` shape function gradients
    common::Buffer<GpuScalar> mLameCoefficients; ///< `2x|# elements|` 1st and 2nd Lame parameters
    GpuScalar mDetHZero;                         ///< Numerical zero for hessian determinant check

    common::Buffer<GpuIndex>
        mVertexTetrahedronPrefix; ///< Vertex-tetrahedron adjacency list's prefix sum
    common::Buffer<GpuIndex>
        mVertexTetrahedronNeighbours; ///< Vertex-tetrahedron adjacency list's neighbour list
    common::Buffer<GpuIndex> mVertexTetrahedronLocalVertexIndices; ///< Vertex-tetrahedron adjacency
                                                                   ///< list's ilocal property

    GpuScalar mRayleighDamping;                         ///< Rayleigh damping coefficient
    GpuScalar mCollisionPenalty;                        ///< Collision penalty coefficient
    GpuScalar mFrictionCoefficient;                     ///< Coefficient of friction
    GpuScalar mSmoothFrictionRelativeVelocityThreshold; ///< IPC smooth friction transition
                                                        ///< function's relative velocity threshold

    GpuIndexVectorX mPptr; ///< `|#partitions+1|` partition pointers, s.t. the range `[Pptr[p],
                           ///< Pptr[p+1])` indexes into `Padj` vertices from partition `p`
    common::Buffer<GpuIndex> mPadj; ///< Partition vertices

    EInitializationStrategy
        mInitializationStrategy;  ///< Strategy to use to determine the initial BCD iterate
    GpuIndex mGpuThreadBlockSize; ///< Number of threads per CUDA thread block
    cuda::stream_t mStream;       ///< Cuda stream on which this VBD instance will run

    /**
     * @brief Obtain Backward Euler time stepping minimization problem for device code
     * @param dt Time step
     * @param dt2 Time step squared
     * @return Backward Euler minimization problem for device code
     */
    kernels::BackwardEulerMinimization BdfDeviceParameters(GpuScalar dt, GpuScalar dt2);
};

} // namespace vbd
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_VBD_INTEGRATOR_H
