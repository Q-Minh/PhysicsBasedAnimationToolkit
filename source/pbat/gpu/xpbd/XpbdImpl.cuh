#ifndef PBAT_GPU_XPBD_XPBD_IMPL_CUH
#define PBAT_GPU_XPBD_XPBD_IMPL_CUH

#include "pbat/Aliases.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/geometry/BvhImpl.cuh"
#include "pbat/gpu/geometry/BvhQueryImpl.cuh"
#include "pbat/gpu/geometry/PrimitivesImpl.cuh"

#include <array>
#include <cstddef>
#include <vector>

namespace pbat {
namespace gpu {
namespace xpbd {

class XpbdImpl
{
  public:
    enum EConstraint : int { StableNeoHookean = 0, Collision, NumberOfConstraintTypes };
    static auto constexpr kConstraintTypes = static_cast<int>(EConstraint::NumberOfConstraintTypes);

    using CollisionCandidateType = typename geometry::BvhQueryImpl::OverlapType;
    using ContactPairType        = typename geometry::BvhQueryImpl::NearestNeighbourPairType;

    /**
     * @brief
     * @param V
     * @param T
     */
    XpbdImpl(
        Eigen::Ref<GpuMatrixX const> const& X,
        Eigen::Ref<GpuIndexMatrixX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& F,
        Eigen::Ref<GpuIndexMatrixX const> const& T,
        Eigen::Ref<GpuIndexVectorX const> const& BV,
        Eigen::Ref<GpuIndexVectorX const> const& BF,
        std::size_t nMaxVertexTetrahedronOverlaps,
        std::size_t nMaxVertexTriangleContacts);
    /**
     * @brief
     */
    void PrepareConstraints();
    /**
     * @brief
     * @param dt
     * @param iterations
     * @params substeps
     */
    void Step(GpuScalar dt, GpuIndex iterations, GpuIndex substeps);
    /**
     * @brief
     * @return
     */
    std::size_t NumberOfParticles() const;
    /**
     * @brief
     * @return
     */
    std::size_t NumberOfConstraints() const;
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
     * @param f
     */
    void SetExternalForces(Eigen::Ref<GpuMatrixX const> const& f);
    /**
     * @brief
     * @param minv
     */
    void SetMassInverse(Eigen::Ref<GpuMatrixX const> const& minv);
    /**
     * @brief
     * @param l
     */
    void SetLameCoefficients(Eigen::Ref<GpuMatrixX const> const& l);
    /**
     * @brief
     * @param alpha
     * @param eConstraint
     */
    void SetCompliance(Eigen::Ref<GpuMatrixX const> const& alpha, EConstraint eConstraint);
    /**
     * @brief
     * @param partitions
     */
    void SetConstraintPartitions(std::vector<std::vector<GpuIndex>> const& partitions);
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
     * @return
     */
    common::Buffer<GpuScalar, 3> const& GetVelocity() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar, 3> const& GetExternalForce() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar> const& GetMassInverse() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar> const& GetLameCoefficients() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar> const& GetShapeMatrixInverse() const;
    /**
     * @brief
     * @return
     */
    common::Buffer<GpuScalar> const& GetRestStableGamma() const;
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
    /**
     * @brief
     * @return
     */
    std::vector<common::Buffer<GpuIndex>> const& GetPartitions() const;

    /**
     * @brief Get the vertex-tetrahedron overlaps
     *
     * @return std::vector<CollisionCandidateType>
     */
    std::vector<CollisionCandidateType> GetVertexTetrahedronCollisionCandidates() const;

    /**
     * @brief
     * @return
     */
    std::vector<ContactPairType> GetVertexTriangleContactPairs() const;

  public:
    geometry::PointsImpl X;    ///< Vertex/particle positions
    geometry::SimplicesImpl V; ///< Boundary vertex simplices
    geometry::SimplicesImpl F; ///< Boundary triangle simplices
    geometry::SimplicesImpl T; ///< Tetrahedral simplices

    geometry::BodiesImpl BV; ///< Bodies of vertex simplices
    geometry::BodiesImpl BF; ///< Bodies of triangle simplices
  private:
    geometry::BvhImpl Tbvh;        ///< Tetrahedron bvh
    geometry::BvhImpl Fbvh;        ///< Triangle bvh (over boundary mesh)
    geometry::BvhQueryImpl Vquery; ///< BVH vertex queries

    common::Buffer<GpuScalar, 3> mPositions;      ///< Vertex/particle positions at time t
    common::Buffer<GpuScalar, 3> mVelocities;     ///< Vertex/particle velocities
    common::Buffer<GpuScalar, 3> mExternalForces; ///< Vertex/particle external forces
    common::Buffer<GpuScalar> mMassInverses;      ///< Vertex/particle mass inverses
    common::Buffer<GpuScalar> mLame;              ///< Lame coefficients
    common::Buffer<GpuScalar>
        mShapeMatrixInverses; ///< 3x3x|#elements| array of material shape matrix inverses
    common::Buffer<GpuScalar>
        mRestStableGamma; ///< 1. + mu/lambda, where mu,lambda are Lame coefficients
    std::array<common::Buffer<GpuScalar>, kConstraintTypes>
        mLagrangeMultipliers; ///< "Lagrange" multipliers:
                              ///< lambda[0] -> Stable Neo-Hookean constraint multipliers
                              ///< lambda[1] -> Collision penalty constraint multipliers
    std::array<common::Buffer<GpuScalar>, kConstraintTypes>
        mCompliance; ///< Compliance
                     ///< alpha[0] -> Stable Neo-Hookean constraint compliance
                     ///< alpha[1] -> Collision penalty constraint compliance

    std::vector<common::Buffer<GpuIndex>> mPartitions; ///< Constraint partitions
    GpuScalar mStaticFrictionCoefficient;              ///< Coulomb static friction coefficient
    GpuScalar mDynamicFrictionCoefficient;             ///< Coulomb dynamic friction coefficient
    Eigen::Vector<GpuScalar, 3> Smin, Smax;            ///< Scene bounding box
};

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_XPBD_XPBD_IMPL_CUH