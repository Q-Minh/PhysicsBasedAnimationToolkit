#ifndef PBAT_GPU_IMPL_XPBD_INTEGRATOR_H
#define PBAT_GPU_IMPL_XPBD_INTEGRATOR_H

#include "pbat/Aliases.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"
#include "pbat/gpu/impl/geometry/Bvh.cuh"
#include "pbat/gpu/impl/geometry/Primitives.cuh"
#include "pbat/sim/xpbd/Data.h"
#include "pbat/sim/xpbd/Enums.h"

#include <array>
#include <thrust/future.h>
#include <vector>

namespace pbat {
namespace gpu {
namespace impl {
namespace xpbd {

class Integrator
{
  public:
    using EConstraint                      = pbat::sim::xpbd::EConstraint;
    static auto constexpr kConstraintTypes = static_cast<int>(EConstraint::NumberOfConstraintTypes);

    using Data = pbat::sim::xpbd::Data;

    /**
     * @brief Construct a new Integrator Impl object
     *
     * @param data
     */
    Integrator(Data const& data);
    /**
     * @brief
     * @param dt
     * @param iterations
     * @param substeps
     */
    void Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps);
    /**
     * @brief
     * @param alpha
     * @param eConstraint
     */
    void SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint);
    /**
     * @brief
     * @param muS
     * @param muK
     */
    void SetFrictionCoefficients(GpuScalar muS, GpuScalar muK);
    /**
     * @brief
     * @param min
     * @param max
     */
    void SetSceneBoundingBox(
        Eigen::Vector<GpuScalar, 3> const& min,
        Eigen::Vector<GpuScalar, 3> const& max);
    /**
     * @brief
     * @param eConstraint
     * @return
     */
    common::Buffer<GpuScalar> const& GetLagrangeMultiplier(EConstraint eConstraint) const;
    /**
     * @brief
     * @param eConstraint
     * @return
     */
    common::Buffer<GpuScalar> const& GetCompliance(EConstraint eConstraint) const;

    // Ideally, these would not be public, but nvcc will otherwise report that
    // "The enclosing parent function ("[...]") for an extended __device__ lambda
    // cannot have private or protected access within its class"
    void ProjectBlockNeoHookeanConstraints(thrust::device_event& e, Scalar dt, Scalar dt2);
    void ProjectClusteredBlockNeoHookeanConstraints(thrust::device_event& e, Scalar dt, Scalar dt2);

  public:
    common::Buffer<GpuScalar, 3> x; ///< Vertex/particle positions
    common::Buffer<GpuIndex> V;     ///< Boundary vertices
    common::Buffer<GpuIndex, 3> F;  ///< Boundary triangles
    common::Buffer<GpuIndex, 4> T;  ///< Tetrahedral simplices
    common::Buffer<GpuIndex> B;     ///< Bodies of boundary vertices

    geometry::Bvh Tbvh; ///< (Line segment + tetrahedron) bvh
    geometry::Bvh Fbvh; ///< Triangle bvh (over boundary mesh)

    common::Buffer<GpuScalar, 3> xt;   ///< Vertex/particle positions at time t
    common::Buffer<GpuScalar, 3> xb;   ///< Vertex/particle positions buffer
    common::Buffer<GpuScalar, 3> v;    ///< Vertex/particle velocities
    common::Buffer<GpuScalar, 3> aext; ///< Vertex/particle external forces
    common::Buffer<GpuScalar> minv;    ///< Vertex/particle mass inverses
    common::Buffer<GpuScalar> lame;    ///< Lame coefficients
    common::Buffer<GpuScalar> DmInv;   ///< 3x3x|#elements| array of material shape matrix inverses
    common::Buffer<GpuScalar> gamma;   ///< 1. + mu/lambda, where mu,lambda are Lame coefficients
    std::array<common::Buffer<GpuScalar>, kConstraintTypes>
        lagrange; ///< "Lagrange" multipliers:
                  ///< lambda[0] -> Stable Neo-Hookean constraint multipliers
                  ///< lambda[1] -> Collision penalty constraint multipliers
    std::array<common::Buffer<GpuScalar>, kConstraintTypes>
        alpha; ///< Compliance
               ///< alpha[0] -> Stable Neo-Hookean constraint compliance
               ///< alpha[1] -> Collision penalty constraint compliance
    std::array<common::Buffer<GpuScalar>, kConstraintTypes>
        beta; ///< Damping
              ///< beta[0] -> Stable Neo-Hookean constraint damping
              ///< beta[1] -> Collision penalty constraint damping

    std::vector<Index> Pptr;       ///< Constraint partitions' pointers
    common::Buffer<GpuIndex> Padj; ///< Constraint partitions' constraints

    std::vector<Index> SGptr;       ///< Clustered constraint partitions' pointers
    common::Buffer<GpuIndex> SGadj; ///< Clustered constraint partitions' constraints
    common::Buffer<Index> Cptr;     ///< Cluster -> constraint map pointers
    common::Buffer<GpuIndex> Cadj;  ///< Cluster -> constraint map constraints

    common::Buffer<GpuScalar> muC;          ///< Collision vertex penalties
    GpuScalar muS;                          ///< Coulomb static friction coefficient
    GpuScalar muK;                          ///< Coulomb dynamic friction coefficient
    Eigen::Vector<GpuScalar, 3> Smin, Smax; ///< Scene bounding box
};

} // namespace xpbd
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_XPBD_INTEGRATOR_H
