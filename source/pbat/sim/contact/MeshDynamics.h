/**
 * @file MeshDynamics.h
 * @author Quoc-Minh Ton-That (tonthat@gmail.com)
 * @brief Mesh contact dynamics
 * @version 0.1
 * @date 2025-11-12
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_CONTACT_MESHDYNAMICS_H
#define PBAT_SIM_CONTACT_MESHDYNAMICS_H

#include "Constraints.h"
#include "MultiMesh.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Atomic.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/Device.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/graph/DenseAdjacencySet.h"
#include "pbat/io/Archive.h"
#include "pbat/math/linalg/FilterEigenvalues.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/contact/Friction.h"
#include "pbat/sim/contact/Potentials.h"
#include "pbat/sim/contact/ogc/Ogc.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cmath>
#include <new>
#include <tbb/parallel_for.h>
#include <type_traits>
#include <vector>

namespace pbat::sim::contact {

/**
 * @brief Mesh contact dynamics
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
class MeshDynamics
{
  public:
    using ScalarType = TScalar; ///< Scalar type
    using IndexType  = TIndex;  ///< Index type
    /**
     * @brief Per constraint quantities
     * @tparam TDistance Mesh distance function
     */
    template <geometry::CMeshDistance TDistance>
    struct alignas(std::hardware_constructive_interference_size) ConstraintData
    {
        using DistanceType             = TDistance;           ///< Distance function type
        static auto constexpr kStencil = TDistance::kStencil; ///< Number of vertices in the stencil
        static auto constexpr kDims    = TDistance::kDims;    ///< Number of dimensions
        static auto constexpr kDofs    = TDistance::kDofs;    ///< Total degrees of freedom
        TScalar mu{0};                                        ///< Complementarity slack relaxation
        TScalar chat;                                         ///< \f$ c(x_k) - gradc(x_k)^T x_k \f$
        math::linalg::mini::SVector<TScalar, kDofs> gradc;    ///< \f$ \nabla c(x_k) \f$
        /**
         * @brief Compute the Lagrange multiplier for the constraint given the current positions and
         * the smooth-step S(c(x)).
         * @param x `kDofs x 1` stacked positions of the vertices in the stencil
         * @param cstep Smooth-step value S(c(x)) for the constraint value at the current positions
         * @return Lagrange multiplier value
         */
        TScalar LagrangeMultiplier(auto&& x, TScalar cstep = TScalar(1)) const
        {
            return (mu * cstep) / DistanceType{}.Eval(x);
        }
    };
    /**
     * @brief Constraint set container
     * @tparam kStencil Number of nodes the constraint depends on
     */
    template <geometry::CMeshDistance TDistance>
    using ConstraintSet = graph::DenseAdjacencySet<TIndex, ConstraintData<TDistance>>;
    /**
     * @brief Point-point contact constraint container
     */
    struct PointPointContactSet : public ConstraintSet<geometry::PointPointDistance<TScalar>>
    {
        using ConstraintDataType     = ConstraintData<geometry::PointPointDistance<TScalar>>;
        using ConstraintFunctionType = geometry::PointPointDistance<TScalar>;
    };
    /**
     * @brief Point-edge contact constraint container
     */
    struct PointEdgeContactSet : public ConstraintSet<geometry::PointEdgeDistance<TScalar>>
    {
        using ConstraintDataType     = ConstraintData<geometry::PointEdgeDistance<TScalar>>;
        using ConstraintFunctionType = geometry::PointEdgeDistance<TScalar>;
    };
    /**
     * @brief Point-triangle contact constraint container
     */
    struct PointTriangleContactSet : public ConstraintSet<geometry::PointTriangleDistance<TScalar>>
    {
        using ConstraintDataType     = ConstraintData<geometry::PointTriangleDistance<TScalar>>;
        using ConstraintFunctionType = geometry::PointTriangleDistance<TScalar>;
    };
    /**
     * @brief Edge-edge contact constraint container
     */
    struct EdgeEdgeContactSet : public ConstraintSet<geometry::EdgeEdgeDistance<TScalar>>
    {
        using ConstraintDataType     = ConstraintData<geometry::EdgeEdgeDistance<TScalar>>;
        using ConstraintFunctionType = geometry::EdgeEdgeDistance<TScalar>;
    };
    /**
     * @brief Contact pair stencil
     */
    struct Stencil
    {
        IndexType u, v; ///< Mesh primitive indices (e.g., vertex index for point, half-edge index
                        ///< for edge, face index for triangle)
        int gu, gv;     ///< Geometry type id (e.g. from OgcState::EGeometry)
    };
    /**
     * @brief Mesh dynamics parameters
     */
    struct Params
    {
        using SelfType = typename MeshDynamics<TScalar, TIndex>::Params; ///< Self type

        ogc::Params<TScalar> mOgcParams; ///< OGC parameters
        TScalar epsv{1e-3}; ///< IPC's relative velocity threshold for static to dynamic friction's
                            ///< smooth transition
        TScalar kc{1e3};    ///< OGC contact stiffness parameter, `kc > 0`
        TScalar mu{0.5};    ///< OGC friction coefficient, `mu >= 0`
        TScalar rqstart{0}; ///< Base query radius (larger than contact radius `r`) on which we add
                            ///< a linear function of inertial target distance to initialize the
                            ///< actual query radius
        TScalar betarq{1};  ///< Slope of the linear function of inertial target distance to add to
                            ///< `rqstart` to initialize the actual query radius
        TScalar gamma{1};   ///< Multiple of dynamics hessian curvature in constraint gradient
                            ///< direction for barrier parameter computation
        TScalar dmin{2e-3}; ///< Loose target minimum contact distance
        bool bDeactivate{false}; ///< Whether to deactivate contacts

        /**
         * @brief Set the OGC parameters
         * @param params OGC parameters
         * @return Reference to this
         */
        SelfType& WithOgcParams(ogc::Params<TScalar> const& params);
        /**
         * @brief Set the frictional contact parameters
         *
         * @param mu Friction coefficient
         * @param epsv Relative velocity threshold for static to dynamic friction's smooth
         * transition
         * @return Reference to this
         */
        SelfType& WithFrictionalContact(TScalar mu, TScalar epsv);
        /**
         * @brief Set the normal contact parameters
         * @param kc Contact stiffness parameter, `kc > 0`
         * @return Reference to this
         */
        SelfType& WithNormalContact(TScalar kc);
        /**
         * @brief Set the query radius initialization parameters
         * @param rqstart Base query radius
         * @param betarq Slope of the linear function of inertial target distance to add to
         * `rqstart` to initialize the actual query radius
         * @return Reference to this
         */
        SelfType& WithQueryRadiusInitialization(TScalar rqstart, TScalar betarq);
        /**
         * @brief Construct the Params object
         * @param bValidate Whether to validate parameters
         * @return Reference to this
         */
        SelfType& Construct(bool bValidate = true);
        /**
         * @brief Query the contact query radius given the inertial target distance
         * @param inertialTargetDistance Inertial target distance (or other relevant distance)
         * @post `mOgcParams.rq` is set
         */
        void ComputeQueryRadius(TScalar inertialTargetDistance);
        /**
         * @brief Activate or deactivate contacts
         * @param bActive true to activate, false to deactivate
         */
        void Activate(bool bActive = true);
        /**
         * @brief Deactivate contacts
         */
        void Deactivate();
        /**
         * @brief Serialize to archive
         * @param archive Archive to serialize to
         */
        void Serialize(io::Archive& archive) const;
        /**
         * @brief Deserialize from archive
         * @param archive Archive to deserialize from
         */
        void Deserialize(io::Archive const& archive);

        /**
         * @brief Read/Write members, not part of the configuration
         */
        TScalar kcp; ///< `kcp = tau*kc*(tau - r)^2`, where `tau = r/2`
        TScalar b;   ///< `b = kc/2*(r - tau)^2 + kcp*log(tau)`, where `tau = r/2`
    };

    /**
     * @brief Construct a new Mesh Dynamics object
     */
    MeshDynamics(Params const& params = Params());
    /**
     * @brief Set the contact geometries
     * @param Xdynamic `3 x |# points|` dynamic point positions (column-major: one point per column)
     * @param dynamicMeshes Dynamic mesh contact geometry representation
     * @param Xstatic `3 x |# points|` static point positions (column-major: one point per column)
     * @param staticMeshes Static mesh contact geometry representation
     */
    void Construct(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xdynamic,
        MultiMesh<IndexType> dynamicMeshes,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xstatic,
        MultiMesh<IndexType> staticMeshes);
    /**
     * @brief Initialize the Mesh Dynamics object
     * @param device Device to use for acceleration structures
     * @pre `SetDynamicGeometry()` or `Construct()` has been called
     */
    void Initialize(geometry::Device device);
    /**
     * @brief Truncate displacements from positions to satisfy the computed displacement bounds
     * @tparam TDerivedXkp1 Writeable matrix type
     * @tparam TMask Eigen dense base s.t. TMask::Scalar is convertible to bool
     * @param Xkp1 `3 x |# points|` or `3*|# points| x 1` proposed new point positions
     * @param mask `|# points| x 1` mask of points to ignore (true = ignore, false = process)
     * @return Number of truncated points in this call.
     * @pre `Initialize()` has been called
     * @post Displacements have been truncated to satisfy the computed displacement bounds
     */
    template <class TDerivedXkp1, class TMask>
    Eigen::Index
    RestoreFeasibility(Eigen::MatrixBase<TDerivedXkp1>& Xkp1, Eigen::DenseBase<TMask> const& mask);
    /**
     * @brief Truncate displacements to satisfy the computed displacement bounds
     * @tparam TDerivedXkp1 Writeable matrix type
     * @param Xkp1 `3 x |# points|` or `3*|# points| x 1` proposed new point positions
     * @return Number of truncated points in this call.
     * @pre `Initialize()` has been called
     * @post Displacements have been truncated to satisfy the computed displacement bounds
     */
    template <class TDerivedXkp1>
    Eigen::Index RestoreFeasibility(Eigen::MatrixBase<TDerivedXkp1>& Xkp1);
    /**
     * @brief Truncate displacements to satisfy the computed displacement bounds
     * @tparam TDerivedDxkp1 Writeable matrix type
     * @tparam TMask Eigen dense base s.t. TMask::Scalar is convertible to bool
     * @param Dxkp1 `3 x |# points|` or `3*|# points| x 1` displacements
     * @param mask `|# points| x 1` mask of points to ignore (true = ignore, false = process)
     * @return Number of truncated points in this call.
     * @pre `Initialize()` has been called
     * @post Displacements have been truncated to satisfy the computed displacement bounds
     */
    template <class TDerivedDxkp1, class TMask>
    Eigen::Index
    MakeStepFeasible(Eigen::MatrixBase<TDerivedDxkp1>& Dxkp1, Eigen::DenseBase<TMask> const& mask);
    /**
     * @brief Request constraint set update
     */
    void RequestConstraintSetUpdate();
    /**
     * @brief Check if constraint set update is required
     * @return true if constraint set update is required, false otherwise
     */
    bool RequiresConstraintSetUpdate() const;
    /**
     * @brief Executes a collision detection pass and computes resulting per-point displacement
     * bounds.
     * @tparam TDerivedX Matrix type
     * @param X `3 x |# points|` current point positions (column-major: one point per column)
     * @pre `RestoreFeasibility()` has been called
     */
    template <class TDerivedX>
    void UpdateConstraintSet(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Linearize all contact constraints, initializing them if necessary.
     * ­@tparam TDerivedx Matrix type
     * @param x `3 x |# points|` current point positions (column-major: one point per column)
     */
    template <class TDerivedx>
    void LinearizeConstraints(Eigen::DenseBase<TDerivedx> const& x);
    /**
     * @brief Compute barrier parameters for all constraints based on an estimate of the objective
     * function's hessian at the specified (through Params) desired minimal contact separation.
     *
     * @tparam TDerivedH Matrix type for the hessian estimate
     * @param H Hessian estimate at the desired minimal contact separation, used to compute a
     * heuristic barrier parameter.
     */
    template <class TDerivedH>
    void ComputeBarrierParameters(Eigen::SparseCompressedBase<TDerivedH> const& H);
    /**
     * @brief Iterate over contacts [cstart, cend) in the contact set
     * @tparam FOnContact Callable type with signature `template <class TConstraintData>
     * void(TConstraintData& C, Stencil const& stencil)` where
     * `std::is_same_v<TConstraintData, typename TContactSet::ConstraintDataType>`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to iterate over
     * @param fOnContact Callback for each contact
     * @param cstart Start index for the contact set
     * @param cend End index for the contact set
     * @note Use this lower-level contact block visitor for parallel processing
     */
    template <class FOnContact, class TContactSet>
    void
    ForEachContact(TContactSet& contactSet, FOnContact&& fOnContact, TIndex cstart, TIndex cend);
    /**
     * @brief Iterate over all contacts in the contact set
     * @tparam FOnContact Callable type with signature `template <class TConstraintData>
     * void(TConstraintData& C, Stencil stencil)` where
     * `std::is_same_v<TConstraintData, typename TContactSet::ConstraintDataType>`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to iterate over
     * @param fOnContact Callback for each contact
     */
    template <class FOnContact, class TContactSet>
    void ForEachContact(TContactSet& contactSet, FOnContact&& fOnContact);
    /**
     * @brief Iterate over all contacts
     * @tparam FOnContact Callable type with signature `template <class TConstraintData>
     * void(TConstraintData& C, Stencil const& stencil)` where
     * `std::is_same_v<TConstraintData, typename TContactSet::ConstraintDataType>`
     * @param fOnContact Callback for each contact
     * @param bParallel Whether to visit all contacts in parallel
     */
    template <class FOnContact>
    void ForAllContacts(FOnContact&& fOnContact, bool bParallel = false);
    /**
     * @brief Load the stencil for a contact pair
     * @param u First mesh primitive index
     * @param v Second mesh primitive index
     * @param gu First mesh geometry index
     * @param gv Second mesh geometry index
     * @return The pair (X, nodes) of stencil (per-column) point matrix and corresponding indices
     * where `X` is a `math::linalg::mini::SMatrix<TScalar, kDims, kStencil>`
     */
    template <class TDerivedx, class TContactSet>
    auto LoadStencil(Eigen::DenseBase<TDerivedx> const& x, Stencil const& stencil);
    /**
     * @brief Load the stencil point indices for a contact pair (index-only, no position loading)
     * @param stencil Contact stencil
     * @return The stencil point indices as `std::array<TIndex, kStencil>`
     */
    template <class TContactSet>
    auto LoadStencil(Stencil const& stencil);
    /**
     * @brief Get the number of truncated points from the last `RestoreFeasibility()`
     * call
     * @return Number of truncated points
     */
    Eigen::Index NumTruncatedPoints() const;
    /**
     * @brief Set the static geometry
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param meshes Mesh contact geometry representation
     */
    void SetStaticGeometry(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        MultiMesh<IndexType> meshes);
    /**
     * @brief Set the dynamic geometry
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param meshes Mesh contact geometry representation
     */
    void SetDynamicGeometry(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        MultiMesh<IndexType> meshes);
    /**
     * @brief Compute contact energies (potential, gradient, hessian) from current positions
     * @param x `3 x |# points|` or `3*|# points| x 1` point positions
     * @param xt `3 x |# points|` or `3*|# points| x 1` point positions s.t. `\Dot{x} = (x - xt) /
     * h`
     * @param h Time step size
     * @param computationFlags Flags indicating which quantities to compute
     */
    template <class TDerivedX, class TDerivedXt>
    void ComputeEnergies(
        Eigen::MatrixBase<TDerivedX> const& x,
        Eigen::MatrixBase<TDerivedXt> const& xt,
        ScalarType h,
        int computationFlags);
    /**
     * @brief Compute the total potential energy from all contacts
     * @return Total potential energy
     * @pre `ComputeEnergies()` has been called with the `Potential` flag
     */
    ScalarType Potential() const;
    /**
     * @brief Compute the total contact gradient
     * @return Total contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    auto Gradient() const -> Eigen::Vector<ScalarType, Eigen::Dynamic>;
    /**
     * @brief Compute the total contact gradient and add it to `g`
     * @tparam TDerivedg Writeable matrix type
     * @param g `3*|# points| x 1` or `3 x |# points|` total contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    template <class TDerivedg>
    void ToGradient(Eigen::MatrixBase<TDerivedg>& g) const;
    /**
     * @brief Compute the normal contact gradient
     * @return Normal contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    auto NormalGradient() const -> Eigen::Vector<ScalarType, Eigen::Dynamic>;
    /**
     * @brief Compute the normal contact gradient and add it to `g`
     * @tparam TDerivedg Writeable matrix type
     * @param g `3*|# points| x 1` or `3 x |# points|` normal contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    template <class TDerivedg>
    void ToNormalGradient(Eigen::MatrixBase<TDerivedg>& g) const;
    /**
     * @brief Compute the frictional contact gradient
     * @return Frictional contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    auto FrictionalGradient() const -> Eigen::Vector<ScalarType, Eigen::Dynamic>;
    /**
     * @brief Compute the frictional contact gradient and add it to `g`
     * @tparam TDerivedg Writeable matrix type
     * @param g `3*|# points| x 1` or `3 x |# points|` frictional contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    template <class TDerivedg>
    void ToFrictionalGradient(Eigen::MatrixBase<TDerivedg>& g) const;
    /**
     * @brief Get the Params object
     * @return Reference to the parameters
     */
    Params const& GetParams() const { return mParams; }
    /**
     * @brief Get the Params object
     * @return Reference to the parameters
     */
    Params& GetParams() { return mParams; }
    /**
     * @brief Get the static point positions
     * @return `3 x |# points|` static point positions
     */
    auto StaticPointPositions() const -> Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const&
    {
        return mXstatic;
    }
    /**
     * @brief Get the Dynamic Meshes object
     * @return MultiMesh<IndexType> const&
     */
    auto DynamicMeshes() const -> MultiMesh<IndexType> const& { return mDynamicMeshes; }
    /**
     * @brief Get the Static Meshes object
     * @return MultiMesh<IndexType> const&
     */
    auto StaticMeshes() const -> MultiMesh<IndexType> const& { return mStaticMeshes; }
    /**
     * @brief Get the Ogc Input object
     * @return ogc::Input<ScalarType, IndexType> const&
     */
    auto OgcInput() const -> ogc::Input<ScalarType, IndexType> const& { return mOgcInput; }
    /**
     * @brief Get the Ogc Input object
     * @return ogc::Input<ScalarType, IndexType>&
     */
    auto OgcInput() -> ogc::Input<ScalarType, IndexType>& { return mOgcInput; }
    /**
     * @brief Get the Ogc State object
     * @return ogc::State<ScalarType, IndexType> const&
     */
    auto OgcState() const -> ogc::State<ScalarType, IndexType> const& { return mOgcState; }
    /**
     * @brief Get the Ogc State object
     * @return ogc::State<ScalarType, IndexType>&
     */
    auto OgcState() -> ogc::State<ScalarType, IndexType>& { return mOgcState; }
    /**
     * @brief Get the point-point contact adjacency set
     * @return Reference to the point-point contact adjacency set
     */
    auto PointPointContacts() const -> PointPointContactSet const& { return mPointPointContacts; }
    /**
     * @brief Get the point-point contact adjacency set
     * @return Reference to the point-point contact adjacency set
     */
    auto PointPointContacts() -> PointPointContactSet& { return mPointPointContacts; }
    /**
     * @brief Get the point-edge contact adjacency set
     * @return Reference to the point-edge contact adjacency set
     */
    auto PointEdgeContacts() const -> PointEdgeContactSet const& { return mPointEdgeContacts; }
    /**
     * @brief Get the point-edge contact adjacency set
     * @return Reference to the point-edge contact adjacency set
     */
    auto PointEdgeContacts() -> PointEdgeContactSet& { return mPointEdgeContacts; }
    /**
     * @brief Get the point-triangle contact adjacency set
     * @return Reference to the point-triangle contact adjacency set
     */
    auto PointTriangleContacts() const -> PointTriangleContactSet const&
    {
        return mPointTriangleContacts;
    }
    /**
     * @brief Get the point-triangle contact adjacency set
     * @return Reference to the point-triangle contact adjacency set
     */
    auto PointTriangleContacts() -> PointTriangleContactSet& { return mPointTriangleContacts; }
    /**
     * @brief Get the edge-edge contact adjacency set
     * @return Reference to the edge-edge contact adjacency set
     */
    auto EdgeEdgeContacts() const -> EdgeEdgeContactSet const& { return mEdgeEdgeContacts; }
    /**
     * @brief Get the edge-edge contact adjacency set
     * @return Reference to the edge-edge contact adjacency set
     */
    auto EdgeEdgeContacts() -> EdgeEdgeContactSet& { return mEdgeEdgeContacts; }
    /**
     * @brief Serialize to archive
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive);
    /**
     * @brief Deserialize from archive
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive const& archive);

  protected:
    /**
     * @brief Transfer OGC contact pairs to our contact sets
     */
    void UpdateContactSetsFromOgcPairs();
    /**
     * @brief Get the geometry prefix arrays for the contact set
     * @return The pair (prefu, prefv)
     */
    template <class TContactSet>
    auto GeometryPrefixArrays();
    /**
     * @brief Load a point's position from the mesh state
     * @param i Point index
     * @param g Geometry type
     * @param xi Output position vector
     * @return The point index on the corresponding geometry g
     */
    template <class TDerivedx>
    auto LoadPoint(Eigen::DenseBase<TDerivedx> const& x, TIndex i, int g, auto&& xi);
    /**
     * @brief Load a half-edge's positions from the mesh state
     * @param he Half-edge index
     * @param g Geometry type
     * @param xi Output position vector for the incoming vertex
     * @param xj Output position vector for the outgoing vertex
     * @return The half-edge point indices (i, j) on the corresponding geometry g
     */
    template <class TDerivedx>
    auto LoadHalfEdge(Eigen::DenseBase<TDerivedx> const& x, TIndex he, int g, auto&& xi, auto&& xj);
    /**
     * @brief Load a triangle's positions from the mesh state
     * @param f Triangle index
     * @param g Geometry type
     * @param xi Output position vector for the first vertex
     * @param xj Output position vector for the second vertex
     * @param xk Output position vector for the third vertex
     * @return The triangle point indices (i, j, k) on the corresponding geometry g
     */
    template <class TDerivedx>
    auto LoadTriangle(
        Eigen::DenseBase<TDerivedx> const& x,
        TIndex f,
        int g,
        auto&& xi,
        auto&& xj,
        auto&& xk);
    /**
     * @brief Load a point's index from the mesh state (index-only, no position loading)
     * @param i Point index
     * @param g Geometry type
     * @return The point index on the corresponding geometry g
     */
    auto LoadPoint(TIndex i, int g);
    /**
     * @brief Load a half-edge's point indices from the mesh state (index-only, no position
     * loading)
     * @param he Half-edge index
     * @param g Geometry type
     * @return The half-edge point indices (i, j) on the corresponding geometry g
     */
    auto LoadHalfEdge(TIndex he, int g);
    /**
     * @brief Load a triangle's point indices from the mesh state (index-only, no position loading)
     * @param f Triangle index
     * @param g Geometry type
     * @return The triangle point indices (i, j, k) on the corresponding geometry g
     */
    auto LoadTriangle(TIndex f, int g);
    /**
     * @brief Compute the curvature \f$ \nabla c^T H \nabla c \f$ of a sparse matrix H in the
     * constraint gradient direction.
     * @tparam TDerivedH Sparse matrix type
     * @tparam kDims Number of spatial dimensions
     * @tparam kStencil Number of nodes in the stencil
     * @param H Sparse matrix
     * @param gradc Constraint gradient vector of size `kStencil * kDims`
     * @param nodes Stencil node indices
     * @return The curvature value \f$ \nabla c^T H \nabla c \f$
     */
    template <class TDerivedH, int kDims, int kStencil>
    TScalar RayleighQuotient(
        Eigen::SparseCompressedBase<TDerivedH> const& H,
        math::linalg::mini::SVector<TScalar, kStencil * kDims> const& gradc,
        std::array<TIndex, kStencil> const& nodes);

  private:
    Params mParams; ///< Mesh dynamics parameters

    /**
     * @brief Contact detection data structures and algorithms
     */
    Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> mXdynamic; ///< Dynamic point position storage
    Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> mXstatic;  ///< Static point position storage
    MultiMesh<IndexType> mDynamicMeshes;                    ///< Dynamic geometry
    MultiMesh<IndexType> mStaticMeshes;                     ///< Static geometry
    ogc::Input<ScalarType, IndexType> mOgcInput;            ///< OGC input data structures
    ogc::State<ScalarType, IndexType> mOgcState; ///< OGC transient algorithm data structures
    Eigen::Index mNumTruncatedPoints{0};         ///< Number of truncated points in last truncation
    bool mRequiresBoundsRecomputation{true};     ///< Whether bounds recomputation is required

    /**
     * @brief Contact constraint sets
     */
    PointPointContactSet mPointPointContacts;       ///< Point-point contact set
    PointEdgeContactSet mPointEdgeContacts;         ///< Point-edge contact set
    PointTriangleContactSet mPointTriangleContacts; ///< Point-triangle contact set
    EdgeEdgeContactSet mEdgeEdgeContacts;           ///< (Half-)Edge-(half-)edge contact set
};

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithOgcParams(ogc::Params<TScalar> const& params)
{
    mOgcParams = params;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithFrictionalContact(TScalar mu, TScalar epsv)
{
    this->mu   = mu;
    this->epsv = epsv;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithNormalContact(TScalar kc)
{
    this->kc = kc;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithQueryRadiusInitialization(
    TScalar _rqstart,
    TScalar _betarq)
{
    this->rqstart = _rqstart;
    this->betarq  = _betarq;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::Construct(bool bValidate)
{
    if (bValidate)
    {
        if (kc <= TScalar(0))
        {
            throw std::invalid_argument("MeshDynamics::Params::Construct(): kc must be positive.");
        }
        if (mu < TScalar(0))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): mu must be non-negative.");
        }
        if (betarq < TScalar(0))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): betarq must be non-negative.");
        }
    }
    auto tau = TScalar(0.5) * mOgcParams.r;
    auto r   = mOgcParams.r;
    kcp      = tau * kc * (tau - r) * (tau - r);
    b        = (TScalar(0.5) * kc) * (r - tau) * (r - tau) + kcp * std::log(tau);
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void
MeshDynamics<TScalar, TIndex>::Params::ComputeQueryRadius(TScalar inertialTargetDistance)
{
    mOgcParams.rq = std::max(mOgcParams.r, rqstart) + betarq * inertialTargetDistance;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::Activate(bool bActive)
{
    this->bDeactivate = not bActive;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::Deactivate()
{
    bDeactivate = true;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::Serialize(io::Archive& archive) const
{
    auto grp = archive.GetOrCreateGroup("pbat.sim.contact.MeshDynamics.Params");
    {
        auto paramsGrp = grp["mOgcParams"];
        mOgcParams.Serialize(paramsGrp);
    }
    grp.WriteMetaData("epsv", epsv);
    grp.WriteMetaData("kc", kc);
    grp.WriteMetaData("mu", mu);
    grp.WriteMetaData("rqstart", rqstart);
    grp.WriteMetaData("betarq", betarq);
    grp.WriteMetaData("kcp", kcp);
    grp.WriteMetaData("b", b);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::Deserialize(io::Archive const& archive)
{
    auto grp = archive["pbat.sim.contact.MeshDynamics.Params"];
    mOgcParams.Deserialize(grp["mOgcParams"]);
    if (grp.HasMetaData("epsv"))
        epsv = grp.ReadMetaData<TScalar>("epsv");
    if (grp.HasMetaData("kc"))
        kc = grp.ReadMetaData<TScalar>("kc");
    if (grp.HasMetaData("mu"))
        mu = grp.ReadMetaData<TScalar>("mu");
    if (grp.HasMetaData("rqstart"))
        rqstart = grp.ReadMetaData<TScalar>("rqstart");
    if (grp.HasMetaData("betarq"))
        betarq = grp.ReadMetaData<TScalar>("betarq");
    if (grp.HasMetaData("kcp"))
        kcp = grp.ReadMetaData<TScalar>("kcp");
    if (grp.HasMetaData("b"))
        b = grp.ReadMetaData<TScalar>("b");
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline MeshDynamics<TScalar, TIndex>::MeshDynamics(Params const& params) : mParams(params)
{
    mParams.Construct();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Construct(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xdynamic,
    MultiMesh<IndexType> dynamicMeshes,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xstatic,
    MultiMesh<IndexType> staticMeshes)
{
    SetDynamicGeometry(Xdynamic, std::move(dynamicMeshes));
    SetStaticGeometry(Xstatic, std::move(staticMeshes));
    mOgcInput.Construct();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Initialize(geometry::Device device)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.Initialize");
    mOgcState.Initialize(device, mOgcInput, mParams.mOgcParams);
    mNumTruncatedPoints          = 0;
    mRequiresBoundsRecomputation = true;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedXkp1, class TMask>
inline Eigen::Index MeshDynamics<TScalar, TIndex>::RestoreFeasibility(
    Eigen::MatrixBase<TDerivedXkp1>& _Xkp1,
    Eigen::DenseBase<TMask> const& mask)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.RestoreFeasibility");
    static_assert(
        std::is_convertible_v<typename TMask::Scalar, bool>,
        "Mask scalar type must be convertible to bool");
    Eigen::Index nTruncated{0};
    auto const nVertices = mDynamicMeshes.V.size();
    auto Xkp1            = _Xkp1.derived().reshaped(3, _Xkp1.size() / 3);
    tbb::parallel_for(Eigen::Index{0}, nVertices, [&](Eigen::Index v) {
        IndexType i          = mDynamicMeshes.V(v);
        bool const bIsMasked = static_cast<bool>(mask(i));
        if (bIsMasked)
            return;
        ScalarType const b             = mOgcState.bv(v);
        auto xk                        = mXdynamic.col(i);
        auto xkp1                      = Xkp1.col(i);
        Eigen::Vector<ScalarType, 3> d = xkp1 - xk;
        ScalarType const dnorm         = d.norm();
        bool const bIsWithinBound      = dnorm <= b;
        if (bIsWithinBound)
            return;
        // x^{k+1} = x^k + (d/|d|)*b = x^k + d * (b/|d|)
        d *= (b / dnorm);
        xkp1 = xk + d;
        common::AtomicAdd(nTruncated, Eigen::Index{1});
    });
    mNumTruncatedPoints += nTruncated;
    mRequiresBoundsRecomputation = mNumTruncatedPoints >= mParams.mOgcParams.gammae * nVertices;
    return nTruncated;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedXkp1>
inline Eigen::Index
MeshDynamics<TScalar, TIndex>::RestoreFeasibility(Eigen::MatrixBase<TDerivedXkp1>& Xkp1)
{
    auto mask = Eigen::Vector<bool, Eigen::Dynamic>::Constant(Xkp1.cols(), false);
    return RestoreFeasibility(Xkp1, mask);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedDxkp1, class TMask>
inline Eigen::Index MeshDynamics<TScalar, TIndex>::MakeStepFeasible(
    Eigen::MatrixBase<TDerivedDxkp1>& _Dxkp1,
    Eigen::DenseBase<TMask> const& mask)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.MakeStepFeasible");
    static_assert(
        std::is_convertible_v<typename TMask::Scalar, bool>,
        "Mask scalar type must be convertible to bool");
    Eigen::Index nTruncated{0};
    auto const nVertices = mDynamicMeshes.V.size();
    auto Dxkp1           = _Dxkp1.derived().reshaped(3, _Dxkp1.size() / 3);
    tbb::parallel_for(Eigen::Index{0}, nVertices, [&](Eigen::Index v) {
        IndexType i          = mDynamicMeshes.V(v);
        bool const bIsMasked = static_cast<bool>(mask(i));
        if (bIsMasked)
            return;
        ScalarType const b        = mOgcState.bv(v);
        auto d                    = Dxkp1.col(i);
        ScalarType const dnorm    = d.norm();
        bool const bIsWithinBound = dnorm <= b;
        if (bIsWithinBound)
            return;
        // x^{k+1} = x^k + (d/|d|)*b = x^k + d * (b/|d|)
        d *= (b / dnorm);
        common::AtomicAdd(nTruncated, Eigen::Index{1});
    });
    mNumTruncatedPoints += nTruncated;
    mRequiresBoundsRecomputation = mNumTruncatedPoints >= mParams.mOgcParams.gammae * nVertices;
    return nTruncated;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::RequestConstraintSetUpdate()
{
    mRequiresBoundsRecomputation = true;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline bool MeshDynamics<TScalar, TIndex>::RequiresConstraintSetUpdate() const
{
    return mRequiresBoundsRecomputation;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedX>
inline void MeshDynamics<TScalar, TIndex>::UpdateConstraintSet(Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.UpdateConstraintSet");
    if (mParams.bDeactivate)
    {
        mOgcState.bv.setConstant(std::numeric_limits<ScalarType>::max());
        mPointPointContacts.Clear();
        mPointEdgeContacts.Clear();
        mPointTriangleContacts.Clear();
        mEdgeEdgeContacts.Clear();
        mRequiresBoundsRecomputation = false;
        mNumTruncatedPoints          = 0;
        return;
    }
    mXdynamic = X.derived();
    mOgcState.PrepareForExecution(mOgcInput, mParams.mOgcParams);
    ogc::Execute(mOgcInput, mParams.mOgcParams, mOgcState);
    UpdateContactSetsFromOgcPairs();
    mRequiresBoundsRecomputation = false;
    mNumTruncatedPoints          = 0;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx>
inline void
MeshDynamics<TScalar, TIndex>::LinearizeConstraints(Eigen::DenseBase<TDerivedx> const& x)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.LinearizeConstraints");
    ForAllContacts(
        [&]<class TConstraintData>(TConstraintData& C, Stencil const& stencil) {
            using DistanceType = typename TConstraintData::DistanceType;
            DistanceType d{};
            auto const [X, _] = LoadStencil(x, stencil);
            auto x            = Reshape<TConstraintData::kDofs, 1>(X);
            C.gradc           = d.Gradient(x);
            C.c               = d.Eval(x) - Dot(C.gradc, x);
        },
        true /*bParallel*/);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedH>
inline void MeshDynamics<TScalar, TIndex>::ComputeBarrierParameters(
    Eigen::SparseCompressedBase<TDerivedH> const& H)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.ComputeBarrierParameters");
    TScalar gamma = mParams.gamma;
    TScalar d2    = mParams.dmin * mParams.dmin;
    ForAllContacts(
        [&]<class TConstraintData>(TConstraintData& C, Stencil const& stencil) {
            using DistanceType = typename TConstraintData::DistanceType;
            std::array<TIndex, TConstraintData::kStencil> nodes = LoadStencil(stencil);
            TScalar const Q =
                RayleighQuotient<TDerivedH, TConstraintData::kDims, TConstraintData::kStencil>(
                    H,
                    C.gradc,
                    nodes);
            C.mu = gamma * d2 * Q;
        },
        true /*bParallel*/);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachContact(
    TContactSet& set,
    FOnContact&& fOnContact,
    TIndex cstart,
    TIndex cend)
{
    auto const [prefixu, prefixv] = GeometryPrefixArrays<TContactSet>();
    int gu{0}, gv{0};
    for (TIndex c = cstart, uprev = 0; c < cend; ++c)
    {
        auto const [u, v, k] = set.WeightedAdjacency(c);
        // Adjacencies are sorted by (u,v), so we always loop over all v incident on u,
        // until we find the next u, in which case we reset the gv geometry index for v.
        if (u > uprev)
        {
            gv    = 0;
            uprev = u;
        }
        // Keep track of geometry types for u and v
        while (u >= prefixu[gu + 1])
            ++gu;
        while (v >= prefixv[gv + 1])
            ++gv;
        // Visit contact
        using ConstraintDataType = typename TContactSet::ConstraintDataType;
        ConstraintDataType& C    = set.template Data<ConstraintDataType>(k);
        fOnContact(C, Stencil{u, v, gu, gv});
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void
MeshDynamics<TScalar, TIndex>::ForEachContact(TContactSet& contactSet, FOnContact&& fOnContact)
{
    ForEachContact(
        contactSet,
        std::forward<FOnContact>(fOnContact),
        TIndex(0),
        static_cast<TIndex>(contactSet.Size()));
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void MeshDynamics<TScalar, TIndex>::ForAllContacts(FOnContact&& fOnContact, bool bParallel)
{
    if (not bParallel)
    {
        ForEachContact(mPointPointContacts, fOnContact);
        ForEachContact(mPointEdgeContacts, fOnContact);
        ForEachContact(mPointTriangleContacts, fOnContact);
        ForEachContact(mEdgeEdgeContacts, fOnContact);
    }
    else
    {
        tbb::task_group tg;
        unsigned int const nThreads = std::thread::hardware_concurrency();
        auto const fForEachThread   = [&tg, nThreads](auto&& f) {
            for (unsigned int t = 0; t < nThreads; ++t)
                tg.run([f, t]() { f(t); });
        };
        auto const fLaunchKernel = [&]<class TConstraintSet>(TConstraintSet& set) {
            TIndex const nConstraints          = static_cast<TIndex>(set.Size());
            TIndex const nConstraintsPerThread = (nConstraints + nThreads - 1) / nThreads;
            fForEachThread([&](unsigned int t) {
                TIndex const cstart = t * nConstraintsPerThread;
                TIndex const cend   = std::min((t + 1) * nConstraintsPerThread, nConstraints);
                ForEachContact(set, fOnContact, cstart, cend);
            });
        };
        fLaunchKernel(mPointPointContacts);
        fLaunchKernel(mPointEdgeContacts);
        fLaunchKernel(mPointTriangleContacts);
        fLaunchKernel(mEdgeEdgeContacts);
        tg.wait();
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx, class TContactSet>
inline auto MeshDynamics<TScalar, TIndex>::LoadStencil(
    Eigen::DenseBase<TDerivedx> const& x,
    Stencil const& stencil)
{
    using ConstraintDataType = typename TContactSet::ConstraintDataType;
    math::linalg::mini::SMatrix<TScalar, ConstraintDataType::kDims, ConstraintDataType::kStencil> X;
    std::array<TIndex, ConstraintDataType::kStencil> nodes;
    if constexpr (std::is_same_v<TContactSet, PointPointContactSet>)
    {
        nodes[0] = LoadPoint(x.derived(), stencil.u, stencil.gu, X.Col(0));
        nodes[1] = LoadPoint(x.derived(), stencil.v, stencil.gv, X.Col(1));
    }
    else if constexpr (std::is_same_v<TContactSet, PointEdgeContactSet>)
    {
        nodes[0] = LoadPoint(x.derived(), stencil.u, stencil.gu, X.Col(0));
        std::tie(nodes[1], nodes[2]) =
            LoadHalfEdge(x.derived(), stencil.v, stencil.gv, X.Col(1), X.Col(2));
    }
    else if constexpr (std::is_same_v<TContactSet, PointTriangleContactSet>)
    {
        nodes[0] = LoadPoint(x.derived(), stencil.u, stencil.gu, X.Col(0));
        std::tie(nodes[1], nodes[2], nodes[3]) =
            LoadTriangle(x.derived(), stencil.v, stencil.gv, X.Col(1), X.Col(2), X.Col(3));
    }
    else if constexpr (std::is_same_v<TContactSet, EdgeEdgeContactSet>)
    {
        std::tie(nodes[0], nodes[1]) =
            LoadHalfEdge(x.derived(), stencil.u, stencil.gu, X.Col(0), X.Col(1));
        std::tie(nodes[2], nodes[3]) =
            LoadHalfEdge(x.derived(), stencil.v, stencil.gv, X.Col(2), X.Col(3));
    }
    else
    {
        static_assert(false, "Unsupported contact set");
    }
    return std::make_pair(X, nodes);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TContactSet>
inline auto MeshDynamics<TScalar, TIndex>::LoadStencil(Stencil const& stencil)
{
    using ConstraintDataType = typename TContactSet::ConstraintDataType;
    std::array<TIndex, ConstraintDataType::kStencil> nodes;
    if constexpr (std::is_same_v<TContactSet, PointPointContactSet>)
    {
        nodes[0] = LoadPoint(stencil.u, stencil.gu);
        nodes[1] = LoadPoint(stencil.v, stencil.gv);
    }
    else if constexpr (std::is_same_v<TContactSet, PointEdgeContactSet>)
    {
        nodes[0]                     = LoadPoint(stencil.u, stencil.gu);
        std::tie(nodes[1], nodes[2]) = LoadHalfEdge(stencil.v, stencil.gv);
    }
    else if constexpr (std::is_same_v<TContactSet, PointTriangleContactSet>)
    {
        nodes[0]                               = LoadPoint(stencil.u, stencil.gu);
        std::tie(nodes[1], nodes[2], nodes[3]) = LoadTriangle(stencil.v, stencil.gv);
    }
    else if constexpr (std::is_same_v<TContactSet, EdgeEdgeContactSet>)
    {
        std::tie(nodes[0], nodes[1]) = LoadHalfEdge(stencil.u, stencil.gu);
        std::tie(nodes[2], nodes[3]) = LoadHalfEdge(stencil.v, stencil.gv);
    }
    else
    {
        static_assert(false, "Unsupported contact set");
    }
    return nodes;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index MeshDynamics<TScalar, TIndex>::NumTruncatedPoints() const
{
    return mNumTruncatedPoints;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void MeshDynamics<TScalar, TIndex>::SetStaticGeometry(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    MultiMesh<IndexType> meshes)
{
    mXstatic      = X;
    mStaticMeshes = std::move(meshes);
    mOgcInput.WithStaticGeometry(
        mXstatic,
        mStaticMeshes.E,
        mStaticMeshes.F,
        mStaticMeshes.GVHEp,
        mStaticMeshes.GVHEadj,
        mStaticMeshes.GHEF,
        mStaticMeshes.EHE);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void MeshDynamics<TScalar, TIndex>::SetDynamicGeometry(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    MultiMesh<IndexType> meshes)
{
    mXdynamic      = X;
    mDynamicMeshes = std::move(meshes);
    mOgcInput.WithDynamicGeometry(
        mXdynamic,
        mDynamicMeshes.V,
        mDynamicMeshes.F,
        mDynamicMeshes.E,
        mDynamicMeshes.VP,
        mDynamicMeshes.FP,
        mDynamicMeshes.EP,
        mDynamicMeshes.GVHEp,
        mDynamicMeshes.GVHEadj,
        mDynamicMeshes.GHEF,
        mDynamicMeshes.EHE);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline TScalar MeshDynamics<TScalar, TIndex>::Potential() const
{
    ScalarType E{0};
    return E;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::Gradient() const
    -> Eigen::Vector<TScalar, Eigen::Dynamic>
{
    Eigen::Vector<TScalar, Eigen::Dynamic> grad(mXdynamic.size());
    grad.setZero();
    ToGradient(grad);
    return grad;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedg>
inline void
MeshDynamics<TScalar, TIndex>::ToGradient([[maybe_unused]] Eigen::MatrixBase<TDerivedg>& g) const
{
    // auto G                   = g.derived().reshaped(3, g.size() / 3);
    // auto fAccumulateGradient = [&](auto const& energies) {
    //     using math::linalg::mini::ToEigen;
    //     for (auto const& e : energies)
    //         G(Eigen::placeholders::all, e.stencil).reshaped() +=
    //             ToEigen(e.gradEn) + ToEigen(e.gradEf);
    // };
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::NormalGradient() const
    -> Eigen::Vector<TScalar, Eigen::Dynamic>
{
    Eigen::Vector<TScalar, Eigen::Dynamic> grad(mXdynamic.size());
    grad.setZero();
    ToNormalGradient(grad);
    return grad;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedg>
inline void MeshDynamics<TScalar, TIndex>::ToNormalGradient(Eigen::MatrixBase<TDerivedg>& g) const
{
    // auto G                   = g.derived().reshaped(3, g.size() / 3);
    // auto fAccumulateGradient = [&](auto const& energies) {
    //     using math::linalg::mini::ToEigen;
    //     for (auto const& e : energies)
    //         G(Eigen::placeholders::all, e.stencil).reshaped() += ToEigen(e.gradEn);
    // };
    // fAccumulateGradient(mVertexVertexEnergies);
    // fAccumulateGradient(mVertexEdgeEnergies);
    // fAccumulateGradient(mVertexTriangleEnergies);
    // fAccumulateGradient(mEdgeEdgeEnergies);
    // fAccumulateGradient(mVertexEnvironmentEnergies);
    // fAccumulateGradient(mEdgeEnvironmentEnergies);
    // fAccumulateGradient(mTriangleEnvironmentEnergies);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::FrictionalGradient() const
    -> Eigen::Vector<TScalar, Eigen::Dynamic>
{
    Eigen::Vector<TScalar, Eigen::Dynamic> grad(mXdynamic.size());
    grad.setZero();
    ToFrictionalGradient(grad);
    return grad;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedg>
inline void
MeshDynamics<TScalar, TIndex>::ToFrictionalGradient(Eigen::MatrixBase<TDerivedg>& g) const
{
    // auto G                   = g.derived().reshaped(3, g.size() / 3);
    // auto fAccumulateGradient = [&](auto const& energies) {
    //     using math::linalg::mini::ToEigen;
    //     for (auto const& e : energies)
    //         G(Eigen::placeholders::all, e.stencil).reshaped() += ToEigen(e.gradEf);
    // };
    // fAccumulateGradient(mVertexVertexEnergies);
    // fAccumulateGradient(mVertexEdgeEnergies);
    // fAccumulateGradient(mVertexTriangleEnergies);
    // fAccumulateGradient(mEdgeEdgeEnergies);
    // fAccumulateGradient(mVertexEnvironmentEnergies);
    // fAccumulateGradient(mEdgeEnvironmentEnergies);
    // fAccumulateGradient(mTriangleEnvironmentEnergies);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Serialize(io::Archive& archive)
{
    auto grp = archive.GetOrCreateGroup("pbat.sim.contact.MeshDynamics");
    {
        auto paramsGrp = grp["mParams"];
        mParams.Serialize(paramsGrp);
    }
    grp.WriteData("mXdynamic", mXdynamic);
    {
        auto dynamicMeshesGrp = grp["mDynamicMeshes"];
        mDynamicMeshes.Serialize(dynamicMeshesGrp);
    }
    if (mXstatic.size() > 0)
    {
        grp.WriteData("mXstatic", mXstatic);
        {
            auto staticMeshesGrp = grp["mStaticMeshes"];
            mStaticMeshes.Serialize(staticMeshesGrp);
        }
    }
    grp.WriteMetaData("mNumTruncatedPoints", mNumTruncatedPoints);
    grp.WriteMetaData(
        "mRequiresBoundsRecomputation",
        static_cast<int>(mRequiresBoundsRecomputation));
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Deserialize(io::Archive const& archive)
{
    auto grp = archive["pbat.sim.contact.MeshDynamics"];
    mParams.Deserialize(grp["mParams"]);
    mXdynamic =
        grp.ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("mXdynamic");
    mDynamicMeshes.Deserialize(grp["mDynamicMeshes"]);
    if (grp.HasMetaData("mNumTruncatedPoints"))
        mNumTruncatedPoints = grp.ReadMetaData<Eigen::Index>("mNumTruncatedPoints");
    if (grp.HasMetaData("mRequiresBoundsRecomputation"))
        mRequiresBoundsRecomputation =
            static_cast<bool>(grp.ReadMetaData<int>("mRequiresBoundsRecomputation"));
    mOgcInput.WithDynamicGeometry(
        mXdynamic,
        mDynamicMeshes.V,
        mDynamicMeshes.F,
        mDynamicMeshes.E,
        mDynamicMeshes.VP,
        mDynamicMeshes.FP,
        mDynamicMeshes.EP,
        mDynamicMeshes.GVHEp,
        mDynamicMeshes.GVHEadj,
        mDynamicMeshes.GHEF,
        mDynamicMeshes.EHE);
    if (grp.HasData("mXstatic") and grp.HasGroup("mStaticMeshes"))
    {
        mXstatic =
            grp.ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("mXstatic");
        mStaticMeshes.Deserialize(grp["mStaticMeshes"]);
        mOgcInput.WithStaticGeometry(
            mXstatic,
            mStaticMeshes.E,
            mStaticMeshes.F,
            mStaticMeshes.GVHEp,
            mStaticMeshes.GVHEadj,
            mStaticMeshes.GHEF,
            mStaticMeshes.EHE);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::UpdateContactSetsFromOgcPairs()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.UpdateContactSetsFromOgcPairs");
    using OgcStateType = decltype(mOgcState);
    tbb::task_group tg;
    auto const fUpdateContactSet = [&](auto& set, auto& newSet, auto nSourcePrimitives) {
        set.Assign(newSet);
        set.Finalize(nSourcePrimitives);
    };
    auto const nPoints    = mOgcState.mPointGeometryPrefix[OgcStateType::EGeometry::Count];
    auto const nHalfEdges = mOgcState.mHalfEdgeGeometryPrefix[OgcStateType::EGeometry::Count];
    tg.run([&] { fUpdateContactSet(mPointPointContacts, mOgcState.mXX, nPoints); });
    tg.run([&] { fUpdateContactSet(mPointEdgeContacts, mOgcState.mXE, nPoints); });
    tg.run([&] { fUpdateContactSet(mPointTriangleContacts, mOgcState.mXF, nPoints); });
    tg.run([&] { fUpdateContactSet(mEdgeEdgeContacts, mOgcState.mEE, nHalfEdges); });
    tg.wait();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TContactSet>
inline auto MeshDynamics<TScalar, TIndex>::GeometryPrefixArrays()
{
    using ConstraintDataType = typename TContactSet::ConstraintDataType;
    if constexpr (std::is_same_v<TContactSet, PointPointContactSet>)
    {
        return std::make_pair(mOgcState.mPointGeometryPrefix, mOgcState.mPointGeometryPrefix);
    }
    else if constexpr (std::is_same_v<TContactSet, PointEdgeContactSet>)
    {
        return std::make_pair(mOgcState.mPointGeometryPrefix, mOgcState.mHalfEdgeGeometryPrefix);
    }
    else if constexpr (std::is_same_v<TContactSet, PointTriangleContactSet>)
    {
        return std::make_pair(mOgcState.mPointGeometryPrefix, mOgcState.mTriangleGeometryPrefix);
    }
    else if constexpr (std::is_same_v<TContactSet, EdgeEdgeContactSet>)
    {
        return std::make_pair(mOgcState.mHalfEdgeGeometryPrefix, mOgcState.mHalfEdgeGeometryPrefix);
    }
    else
    {
        static_assert(false, "Unsupported contact set");
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx>
inline auto MeshDynamics<TScalar, TIndex>::LoadPoint(
    Eigen::DenseBase<TDerivedx> const& x,
    TIndex i,
    int g,
    auto&& xi)
{
    using EGeometry = decltype(mOgcState)::EGeometry;
    using math::linalg::mini::FromEigen;
    static auto constexpr kDims = std::decay_t<decltype(xi)>::kRows;
    i -= mOgcState.mPointGeometryPrefix[g];
    switch (g)
    {
        case EGeometry::Dynamic: xi = FromEigen(x.col(i).template segment<kDims>()); break;
        case EGeometry::Static: xi = FromEigen(mXstatic.col(i).template segment<kDims>()); break;
        default: break;
    }
    return i;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx>
inline auto MeshDynamics<TScalar, TIndex>::LoadHalfEdge(
    Eigen::DenseBase<TDerivedx> const& x,
    TIndex he,
    int g,
    auto&& xi,
    auto&& xj)
{
    using EGeometry = decltype(mOgcState)::EGeometry;
    using math::linalg::mini::FromEigen;
    static auto constexpr kDims = std::decay_t<decltype(xi)>::kRows;
    he -= mOgcState.mHalfEdgeGeometryPrefix[g];
    TIndex i, j;
    switch (g)
    {
        case EGeometry::Dynamic: {
            i  = geometry::IncomingVertex(*mOgcInput.F, he);
            j  = geometry::OutgoingVertex(*mOgcInput.F, he);
            xi = FromEigen(x.col(i).template segment<kDims>());
            xj = FromEigen(x.col(j).template segment<kDims>());
        }
        break;
        case EGeometry::Static: {
            i  = geometry::IncomingVertex(*mOgcInput.Fenv, he);
            j  = geometry::OutgoingVertex(*mOgcInput.Fenv, he);
            xi = FromEigen(mXstatic.col(i).template segment<kDims>());
            xj = FromEigen(mXstatic.col(j).template segment<kDims>());
        }
        break;
        default: break;
    }
    return std::make_pair(
        mOgcState.mPointGeometryPrefix[g] + i,
        mOgcState.mPointGeometryPrefix[g] + j);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx>
inline auto MeshDynamics<TScalar, TIndex>::LoadTriangle(
    Eigen::DenseBase<TDerivedx> const& x,
    TIndex f,
    int g,
    auto&& xi,
    auto&& xj,
    auto&& xk)
{
    using EGeometry = decltype(mOgcState)::EGeometry;
    using math::linalg::mini::FromEigen;
    static auto constexpr kDims = std::decay_t<decltype(xi)>::kRows;
    f -= mOgcState.mTriangleGeometryPrefix[g];
    TIndex i, j, k;
    switch (g)
    {
        case EGeometry::Dynamic: {
            auto xinds = mOgcInput.F->col(f);
            i          = xinds(0);
            j          = xinds(1);
            k          = xinds(2);
            xi         = FromEigen(x.col(i).template segment<kDims>());
            xj         = FromEigen(x.col(j).template segment<kDims>());
            xk         = FromEigen(x.col(k).template segment<kDims>());
        }
        break;
        case EGeometry::Static: {
            auto xinds = mOgcInput.Fenv->col(f);
            i          = xinds(0);
            j          = xinds(1);
            k          = xinds(2);
            xi         = FromEigen(mXstatic.col(i).template segment<kDims>());
            xj         = FromEigen(mXstatic.col(j).template segment<kDims>());
            xk         = FromEigen(mXstatic.col(k).template segment<kDims>());
        }
        break;
        default: break;
    }
    return std::make_tuple(
        mOgcState.mPointGeometryPrefix[g] + i,
        mOgcState.mPointGeometryPrefix[g] + j,
        mOgcState.mPointGeometryPrefix[g] + k);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::LoadPoint(TIndex i, [[maybe_unused]] int g)
{
    // This is trivial, but we just keep it for API consistency
    return i;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::LoadHalfEdge(TIndex he, int g)
{
    using EGeometry = decltype(mOgcState)::EGeometry;
    he -= mOgcState.mHalfEdgeGeometryPrefix[g];
    TIndex i{}, j{};
    switch (g)
    {
        case EGeometry::Dynamic: {
            i = geometry::IncomingVertex(*mOgcInput.F, he);
            j = geometry::OutgoingVertex(*mOgcInput.F, he);
        }
        break;
        case EGeometry::Static: {
            i = geometry::IncomingVertex(*mOgcInput.Fenv, he);
            j = geometry::OutgoingVertex(*mOgcInput.Fenv, he);
        }
        break;
        default: break;
    }
    return std::make_pair(
        mOgcState.mPointGeometryPrefix[g] + i,
        mOgcState.mPointGeometryPrefix[g] + j);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::LoadTriangle(TIndex f, int g)
{
    using EGeometry = decltype(mOgcState)::EGeometry;
    f -= mOgcState.mTriangleGeometryPrefix[g];
    TIndex i{}, j{}, k{};
    switch (g)
    {
        case EGeometry::Dynamic: {
            auto xinds = mOgcInput.F->col(f);
            i          = xinds(0);
            j          = xinds(1);
            k          = xinds(2);
        }
        break;
        case EGeometry::Static: {
            auto xinds = mOgcInput.Fenv->col(f);
            i          = xinds(0);
            j          = xinds(1);
            k          = xinds(2);
        }
        break;
        default: break;
    }
    return std::make_tuple(
        mOgcState.mPointGeometryPrefix[g] + i,
        mOgcState.mPointGeometryPrefix[g] + j,
        mOgcState.mPointGeometryPrefix[g] + k);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedH, int kDims, int kStencil>
inline TScalar MeshDynamics<TScalar, TIndex>::RayleighQuotient(
    Eigen::SparseCompressedBase<TDerivedH> const& H,
    math::linalg::mini::SVector<TScalar, kStencil * kDims> const& gradc,
    std::array<TIndex, kStencil> const& nodes)
{
    TScalar gcTHgc{0};
    auto const nNodes = static_cast<TIndex>(nodes.size());
    for (TIndex kni = 0; kni < nNodes; ++kni)
    {
        TIndex ni = nodes[kni];
        TIndex ib = kDims * ni;
        for (TIndex ki = 0; ki < kDims; ++ki)
        {
            TIndex i = ib + ki;
            if (i >= H.outerSize())
                continue;
            TIndex knj{0};
            using InnerIteratorType = typename std::remove_cvref_t<decltype(H)>::InnerIterator;
            for (InnerIteratorType it(H, i); it; ++it)
            {
                TIndex nk = it.index() / kDims;
                while (nk >= nodes[knj])
                    ++knj;
                TIndex nj = nodes[knj];
                if (nk == nj)
                {
                    TIndex kj = it.index() % kDims;
                    gcTHgc += gradc(kni * kDims + ki) * it.value() * gradc(knj * kDims + kj);
                }
            }
        }
    }
    TScalar gcTgc = Dot(gradc, gradc);
    return gcTHgc / gcTgc;
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHDYNAMICS_H
