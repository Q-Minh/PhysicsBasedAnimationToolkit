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
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/contact/Friction.h"
#include "pbat/sim/contact/ogc/Ogc.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cmath>
#include <new>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
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
    using SelfType   = MeshDynamics<TScalar, TIndex>; ///< Self type
    using ScalarType = TScalar;                       ///< Scalar type
    using IndexType  = TIndex;                        ///< Index type
    /**
     * @brief Constraint decay value with default 1-initialization semantics
     */
    struct Decay
    {
        TScalar gamma{1}; ///< Ensure 1-initialization
    };
    /**
     * @brief Constraint set container
     * @tparam TDistance Mesh distance function
     */
    template <geometry::CMeshDistance TDistance>
    using ConstraintSet = graph::DenseAdjacencySet<
        TIndex,
        /*0*/ TScalar /*lambda*/,
        /*1*/ TScalar /*s*/,
        /*2*/ Decay /*gamma*/,
        /*3*/ TScalar /*chat*/,
        /*4*/ math::linalg::mini::SVector<TScalar, TDistance::kDofs> /*gradc*/,
        /*5*/ TScalar /*c(x)*/,
        /*6*/ math::linalg::mini::SMatrix<TScalar, 3, 2> /*tangent basis*/,
        /*7*/
        math::linalg::mini::SVector<TScalar, TDistance::kStencil> /*tangential operator weights*/,
        /*8*/
        math::linalg::mini::SVector<TScalar, 2> /*c_f(x) = T^T W (x - xt)*/,
        /*9*/ math::linalg::mini::SVector<TScalar, 2> /*dhat = T^T W xt*/,
        /*10*/ math::linalg::mini::SVector<TScalar, 2> /*lambda_f*/
        >;
    /**
     * @brief Friction data accessor
     * @tparam TContactSet Contact set type (possibly const)
     */
    template <class TContactSet>
    struct FrictionAccessor
    {
        using ContactSetType = std::remove_const_t<TContactSet>; ///< Contact set type
        using ConstraintFunctionType =
            typename ContactSetType::ConstraintFunctionType; ///< Constraint function type
        static auto constexpr kStencil = ConstraintFunctionType::kStencil; ///< Stencil size
        static auto constexpr kDims    = ConstraintFunctionType::kDims;    ///< Dimension size
        static auto constexpr kDofs    = ConstraintFunctionType::kDofs; ///< Degree of freedom size
        TContactSet& set; ///< Reference to the contact set (possibly const)
        TIndex k;         ///< Constraint data index
        /**
         * @brief Tangential basis at the contact point \f$ \mathbf{T} \in \mathbb{R}^{3 \times 2}
         * \f$
         * @return math::linalg::mini::SMatrix<TScalar, 3, 2>& or const&
         */
        auto& TangentBasis() { return set.template Data<6>(k); }
        /**
         * @brief Weights \f$ W \f$ s.t. \f$ u = T^T W v \f$ for stencil
         * velocity \f$ v \f$ and tangent space (2D) velocity \f$ u \f$.
         * @return math::linalg::mini::SVector<TScalar, kStencil>& or const&
         */
        auto& Weights() { return set.template Data<7>(k); }
        /**
         * @brief Friction constraint value \f$ c_f(\mathbf{x}) = \mathbf{T}^T \mathbf{W}
         * (\mathbf{x} - \mathbf{x}_t) \f$
         * @return math::linalg::mini::SVector<TScalar, 2>& or const&
         */
        auto& Eval() { return set.template Data<8>(k); }
        /**
         * @brief Evaluate the friction constraint value
         * @param xc Stencil point positions
         * @return math::linalg::mini::SVector<TScalar, 2>
         */
        auto Eval(auto&& xc) -> math::linalg::mini::SVector<TScalar, 2>
        {
            auto Xc       = Reshape<kDims, kStencil>(xc);
            auto const& T = TangentBasis();
            auto const& W = Weights();
            return T.Transpose() * (Xc * W) - Chat();
        }
        /**
         * @brief \f$ \hat{d} = \mathbf{T}^T \mathbf{W} \mathbf{x}_t \f$
         * @return math::linalg::mini::SVector<TScalar, 2>& or const&
         */
        auto& Chat() { return set.template Data<9>(k); }
        /**
         * @brief Friction Lagrange multiplier \f$ \boldsymbol{\lambda}_f \f$
         * @return math::linalg::mini::SVector<TScalar, 2>& or const&
         */
        auto& Lambda() { return set.template Data<10>(k); }
        /**
         * @brief Compute and store tangent basis and weights from stencil positions.
         * @tparam TDerivedXtc A mini matrix type
         * @tparam TDerivedT A mini matrix type
         * @tparam TDerivedW A mini matrix type
         * @param Xc `kDims x kStencil` stencil point positions
         * @param T `3 x 2` tangent basis matrix
         * @param W `kStencil x 1` weights vector
         */
        template <
            math::linalg::mini::CMatrix Txc,
            math::linalg::mini::CMatrix TT,
            math::linalg::mini::CMatrix TW>
        void ComputeTangentBasisAndWeights(Txc const& xc, TT& T, TW& W)
        {
            static_assert(Txc::kRows == kDofs and Txc::kCols == 1, "Invalid stencil matrix shape");
            static_assert(TT::kRows == 3 and TT::kCols == 2, "Invalid tangent basis matrix shape");
            static_assert(TW::kRows == kStencil and TW::kCols == 1, "Invalid weights vector shape");
            auto Xc = Reshape<kDims, kStencil>(xc);
            using math::linalg::mini::SVector;
            if constexpr (std::is_same_v<TContactSet, PointPointContactSet>)
            {
                T    = PointPointTangentialBasis(Xc.Col(0), Xc.Col(1));
                W(0) = TScalar(1);
                W(1) = -TScalar(1);
            }
            else if constexpr (std::is_same_v<TContactSet, PointEdgeContactSet>)
            {
                auto uv = geometry::ClosestPointQueries::UvPointOnLineSegment(
                    Xc.Col(0),
                    Xc.Col(1),
                    Xc.Col(2));
                SVector<TScalar, 3> xc       = uv(0) * Xc.Col(1) + uv(1) * Xc.Col(2);
                T                            = PointPointTangentialBasis(Xc.Col(0), xc);
                W(0)                         = TScalar(1);
                W.template Slice<2, 1>(1, 0) = -uv;
            }
            else if constexpr (std::is_same_v<TContactSet, PointTriangleContactSet>)
            {
                auto uvw = geometry::ClosestPointQueries::UvwPointInTriangle(
                    Xc.Col(0),
                    Xc.Col(1),
                    Xc.Col(2),
                    Xc.Col(3));
                SVector<TScalar, 3> xc =
                    uvw(0) * Xc.Col(1) + uvw(1) * Xc.Col(2) + uvw(2) * Xc.Col(3);
                T                            = PointPointTangentialBasis(Xc.Col(0), xc);
                W(0)                         = TScalar(1);
                W.template Slice<3, 1>(1, 0) = -uvw;
            }
            else if constexpr (std::is_same_v<TContactSet, EdgeEdgeContactSet>)
            {
                auto st = geometry::ClosestPointQueries::LineSegments(
                    Xc.Col(0),
                    Xc.Col(1),
                    Xc.Col(2),
                    Xc.Col(3));
                SVector<TScalar, 3> xc1 = st(0) * Xc.Col(0) + (1 - st(0)) * Xc.Col(1);
                SVector<TScalar, 3> xc2 = st(1) * Xc.Col(2) + (1 - st(1)) * Xc.Col(3);
                T                       = PointPointTangentialBasis(xc1, xc2);
                W(0)                    = TScalar(1) - st(0);
                W(1)                    = st(0);
                W(2)                    = -(TScalar(1) - st(1));
                W(3)                    = -(st(1));
            }
            else
            {
                static_assert(false, "Unsupported contact set");
            }
        }
        /**
         * @brief Compute the linearization of the constraint at the given point.
         * @param xc `kDofs x 1` stencil node positions
         * @param xtc `kDofs x 1` stencil node positions at time t
         */
        void ComputeLinearization(auto&& xc, auto&& xtc)
        {
            auto& T = TangentBasis();
            auto& W = Weights();
            ComputeTangentBasisAndWeights(xc, T, W);
            auto Xtc = Reshape<kDims, kStencil>(xtc);
            Chat()   = T.Transpose() * Xtc * W;
        }
    };
    /**
     * @brief Lightweight accessor for constraint data in a contact set.
     * @tparam TContactSet Contact set type (possibly const)
     */
    template <class TContactSet>
    struct ConstraintAccessor
    {
        using ContactSetType = std::remove_const_t<TContactSet>; ///< Contact set type
        using ConstraintFunctionType =
            typename ContactSetType::ConstraintFunctionType; ///< Constraint function type
        static auto constexpr kStencil = ConstraintFunctionType::kStencil; ///< Stencil size
        static auto constexpr kDims    = ConstraintFunctionType::kDims;    ///< Dimension size
        static auto constexpr kDofs    = ConstraintFunctionType::kDofs; ///< Degree of freedom size
        TContactSet& set; ///< Reference to the contact set (possibly const)
        TIndex k;         ///< Constraint data index
        /**
         * @brief Get the constraint function for this contact.
         * @return ConstraintFunctionType
         */
        ConstraintFunctionType Function() const { return ConstraintFunctionType{}; }
        /**
         * @brief Compute the linearization of the constraint at the given point.
         * @param xc `kDims x kStencil` stencil node positions
         */
        void ComputeLinearization(auto&& xc)
        {
            ConstraintFunctionType d{};
            Eval() = d.Eval(xc);
            Grad() = d.Gradient(xc);
            Chat() = Eval() - Dot(Grad(), xc);
        }
        /**
         * @brief Lagrange multiplier
         * @return TScalar& or TScalar const&
         */
        auto& Lambda() { return set.template Data<0>(k); }
        /**
         * @brief Inequality slack variable
         * @return TScalar& or TScalar const&
         */
        auto& Slack() { return set.template Data<1>(k); }
        /**
         * @brief Lagrangian and penalty augmentation decay factor
         * @return TScalar& or TScalar const&
         */
        auto& Decay() { return set.template Data<2>(k).gamma; }
        /**
         * @brief \f$ c(\mathbf{x}_k) - \nabla c(\mathbf{x}_k) \cdot \mathbf{x}_k \f$
         * @return TScalar& or TScalar const&
         */
        auto& Chat() { return set.template Data<3>(k); }
        /**
         * @brief Gradient of the constraint at linearized point \f$ \mathbf{x}_k \f$
         * @return math::linalg::mini::SVector<TScalar, kDofs>& or
         * math::linalg::mini::SVector<TScalar, kDofs> const&
         */
        auto& Grad() { return set.template Data<4>(k); }
        /**
         * @brief Last evaluated constraint value
         * @return TScalar& or TScalar const&
         */
        auto& Eval() { return set.template Data<5>(k); }
        /**
         * @brief Evaluates the linearized constraint function at `x`
         * @param x `kDofs x 1` mini::CMatrix
         * @return auto
         */
        auto Eval(auto&& x) { return Chat() + Dot(Grad(), x); }
        /**
         * @brief Friction data accessor
         * @return auto
         */
        auto Friction() { return FrictionAccessor<TContactSet>{set, k}; }
    };
    /**
     * @brief Point-point contact constraint container
     */
    struct PointPointContactSet : public ConstraintSet<geometry::PointPointDistance<TScalar>>
    {
        using ConstraintFunctionType = geometry::PointPointDistance<TScalar>;
        using AccessorType           = ConstraintAccessor<PointPointContactSet>;
        using ConstAccessorType      = ConstraintAccessor<PointPointContactSet const>;
    };
    /**
     * @brief Point-edge contact constraint container
     */
    struct PointEdgeContactSet : public ConstraintSet<geometry::PointEdgeDistance<TScalar>>
    {
        using ConstraintFunctionType = geometry::PointEdgeDistance<TScalar>;
        using AccessorType           = ConstraintAccessor<PointEdgeContactSet>;
        using ConstAccessorType      = ConstraintAccessor<PointEdgeContactSet const>;
    };
    /**
     * @brief Point-triangle contact constraint container
     */
    struct PointTriangleContactSet : public ConstraintSet<geometry::PointTriangleDistance<TScalar>>
    {
        using ConstraintFunctionType = geometry::PointTriangleDistance<TScalar>;
        using AccessorType           = ConstraintAccessor<PointTriangleContactSet>;
        using ConstAccessorType      = ConstraintAccessor<PointTriangleContactSet const>;
    };
    /**
     * @brief Edge-edge contact constraint container
     */
    struct EdgeEdgeContactSet : public ConstraintSet<geometry::EdgeEdgeDistance<TScalar>>
    {
        using ConstraintFunctionType = geometry::EdgeEdgeDistance<TScalar>;
        using AccessorType           = ConstraintAccessor<EdgeEdgeContactSet>;
        using ConstAccessorType      = ConstraintAccessor<EdgeEdgeContactSet const>;
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
        TScalar mu{0.5};    ///< Static friction coefficient, `mu >= 0`
        TScalar rqstart{0}; ///< Base query radius (larger than contact radius `r`) on which we add
                            ///< a linear function of inertial target distance to initialize the
                            ///< actual query radius
        TScalar betarq{1};  ///< Slope of the linear function of inertial target distance to add to
                            ///< `rqstart` to initialize the actual query radius
        TScalar gamma{1};   ///< Multiple of dynamics hessian curvature in constraint gradient
                            ///< direction for penalty parameter computation
        TScalar gammaf{1};  ///< Multiple of dynamics hessian curvature in constraint gradient
                            ///< direction for frictional contact
        TScalar dmin{5e-4}; ///< Loose target minimum contact distance
        TScalar decay{0.9}; ///< Decay factor for constraint deactivation
        TScalar decaylo{0.01};   ///< Decay threshold under which constraints are deactivated
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
         * @brief Set the sequential augmented Lagrangian parameters
         * @param gamma_ Multiple of dynamics hessian curvature for penalty parameter computation
         * @param gammaf_ Same as gamma_ but for frictional contact
         * @param dmin_ Loose target minimum contact distance, `dmin_ > 0`
         * @param decay_ Decay factor for constraint deactivation
         * @param decaylo_ Decay threshold under which constraints are deactivated
         * @return Reference to this
         */
        SelfType& WithSequentialAugmentedLagrangian(
            TScalar gamma_,
            TScalar gammaf_,
            TScalar dmin_,
            TScalar decay_,
            TScalar decaylo_);
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

      private:
        /**
         * @brief Compute the OGC two-stage activation coefficients `kcp` and `b`
         * from the current values of `r` and `kc`.
         * @pre `r > 0`, `kc > 0`
         */
        void UpdateOgcTwoStageActivationCoefficients();
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
     * @param bComputeReverseContactPairs Store reverse contact pair (j,i) for contact pair (i,j) if
     * true
     * @pre `RestoreFeasibility()` has been called
     */
    template <class TDerivedX>
    void UpdateConstraintSet(
        Eigen::DenseBase<TDerivedX> const& X,
        bool bComputeReverseContactPairs = false);
    /**
     * @brief Linearize all contact constraints, initializing them if necessary.
     * @tparam TDerivedx Matrix type
     * @tparam TDerivedxt Matrix type
     * @param x `3 x |# points|` current point positions (column-major: one point per column)
     * @param xt `3 x |# points|` current point positions at time t (column-major: one point per
     * column)
     * @post `Chat()`, `Grad()` and `Eval()` are populated on each constraint
     */
    template <class TDerivedx, class TDerivedxt>
    void LinearizeConstraints(
        Eigen::MatrixBase<TDerivedx> const& x,
        Eigen::MatrixBase<TDerivedxt> const& xt);
    /**
     * @brief Update the penalty parameter for the contact constraints.
     * @tparam TDerived Sparse matrix type
     * @param H Hessian of objective function
     */
    template <class TDerived>
    void UpdatePenaltyParameter(Eigen::SparseCompressedBase<TDerived> const& H);
    /**
     * @brief Enum for dual variable masks
     */
    enum EDualVariable : int { Slack = 1 << 0, LagrangeMultiplier = 1 << 1 };
    /**
     * @brief Update dual variables (slacks, multipliers)
     * @tparam Mask Bitmask for selecting which dual variable(s) to update
     * @tparam TDerivedx Input position matrix type
     * @param x `3 x |# points|` or `3*|# points|` current point positions (column-major: one point
     * per column)¸
     * @post `c(x) = c(x_k) + \nabla c(x_k) \cdot (x - x_k)` for each contact
     */
    template <int Mask, class TDerivedx>
    void UpdateDual(Eigen::MatrixBase<TDerivedx> const& x);
    /**
     * @brief Iterate over all contacts
     * @tparam FOnContact Callable type with signature `template <class TContactSet>
     * void(typename TContactSet::AccessorType C, Stencil stencil, std::int32_t threadId)`
     * @param fOnContact Callback for each contact
     * @param nThreads Number of threads to use for parallel processing. If `nThreads <= 1`,
     * contacts will be processed sequentially.
     */
    template <class FOnContact>
    void ForAllContacts(FOnContact&& fOnContact, std::int32_t nThreads = 1);
    /**
     * @brief Iterate over all contacts
     * @tparam FOnContact Callable type with signature `template <class TContactSet>
     * void(typename TContactSet::ConstAccessorType C, Stencil stencil, std::int32_t threadId)`
     * @param fOnContact Callback for each contact
     * @param nThreads Number of threads to use for parallel processing. If `nThreads <= 1`,
     * contacts will be processed sequentially.
     */
    template <class FOnContact>
    void ForAllContacts(FOnContact&& fOnContact, std::int32_t nThreads = 1) const;
    /**
     * @brief Iterate over all contacts in the contact set
     * @tparam FOnContact Callable type with signature `template <class TContactSet>
     * void(typename TContactSet::AccessorType C, Stencil stencil, std::int32_t threadId)`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to iterate over
     * @param fOnContact Callback for each contact
     * @param nThreads Number of threads to use for parallel processing. If `nThreads <= 1`,
     * contacts will be processed sequentially.
     */
    template <class FOnContact, class TContactSet>
    void
    ForEachContact(TContactSet& contactSet, FOnContact&& fOnContact, std::int32_t nThreads = 1);
    /**
     * @brief Iterate over all contacts in the contact set
     * @tparam FOnContact Callable type with signature `template <class TContactSet>
     * void(typename TContactSet::ConstAccessorType C, Stencil stencil, std::int32_t threadId)`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to iterate over
     * @param fOnContact Callback for each contact
     * @param nThreads Number of threads to use for parallel processing. If `nThreads <= 1`,
     * contacts will be processed sequentially.
     */
    template <class FOnContact, class TContactSet>
    void ForEachContact(
        TContactSet const& contactSet,
        FOnContact&& fOnContact,
        std::int32_t nThreads = 1) const;
    /**
     * @brief Visit all point-point contacts involving point `i`.
     * @tparam FOnContact Callable with signature
     * `void(PointPointContactSet::AccessorType C, Stencil stencil)`
     * @param i Point index
     * @param fOnContact Callback for each contact `(i, j)` and `(j, i)` if reverse contact pairs
     * have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachPointPointContact(IndexType i, FOnContact&& fOnContact);
    /**
     * @brief Visit all point-point contacts involving point `i` (const).
     * @tparam FOnContact Callable with signature
     * `void(PointPointContactSet::ConstAccessorType C, Stencil stencil)`
     * @param i Point index
     * @param fOnContact Callback for each contact `(i, j)` and `(j, i)` if reverse contact pairs
     * have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachPointPointContact(IndexType i, FOnContact&& fOnContact) const;
    /**
     * @brief Visit all point-edge contacts for point `i`.
     * @tparam FOnContact Callable with signature
     * `void(PointEdgeContactSet::AccessorType C, Stencil stencil)`
     * @param i Point index
     * @param fOnContact Callback for each contact `(i, he)` and `(he, i)` if reverse contact pairs
     * have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachPointEdgeContact(IndexType i, FOnContact&& fOnContact);
    /**
     * @brief Visit all point-edge contacts for point `i` (const).
     * @tparam FOnContact Callable with signature
     * `void(PointEdgeContactSet::ConstAccessorType C, Stencil stencil)`
     * @param i Point index
     * @param fOnContact Callback for each contact `(i, he)` and `(he, i)` if reverse contact pairs
     * have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachPointEdgeContact(IndexType i, FOnContact&& fOnContact) const;
    /**
     * @brief Visit all point-edge contacts incident on half-edge `he`.
     * @tparam FOnContact Callable with signature
     * `void(PointEdgeContactSet::AccessorType C, Stencil stencil)`
     * @param he Half-edge index
     * @param fOnContact Callback for each contact `(i, he)`
     * @pre Reverse contact pairs have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachEdgePointContact(IndexType he, FOnContact&& fOnContact);
    /**
     * @brief Visit all point-edge contacts incident on half-edge `he` (const).
     * @tparam FOnContact Callable with signature
     * `void(PointEdgeContactSet::ConstAccessorType C, Stencil stencil)`
     * @param he Half-edge index
     * @param fOnContact Callback for each contact `(i, he)`
     * @pre Reverse contact pairs have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachEdgePointContact(IndexType he, FOnContact&& fOnContact) const;
    /**
     * @brief Visit all point-triangle contacts for point `i`.
     * @tparam FOnContact Callable with signature
     * `void(PointTriangleContactSet::AccessorType C, Stencil stencil)`
     * @param i Point index
     * @param fOnContact Callback for each contact `(i, f)`
     */
    template <class FOnContact>
    void ForEachPointTriangleContact(IndexType i, FOnContact&& fOnContact);
    /**
     * @brief Visit all point-triangle contacts for point `i` (const).
     * @tparam FOnContact Callable with signature
     * `void(PointTriangleContactSet::ConstAccessorType C, Stencil stencil)`
     * @param i Point index
     * @param fOnContact Callback for each contact `(i, f)`
     */
    template <class FOnContact>
    void ForEachPointTriangleContact(IndexType i, FOnContact&& fOnContact) const;
    /**
     * @brief Visit all point-triangle contacts incident on triangle `f`.
     * @tparam FOnContact Callable with signature
     * `void(PointTriangleContactSet::AccessorType C, Stencil stencil)`
     * @param f Triangle index
     * @param fOnContact Callback for each contact `(i, f)`
     * @pre Reverse contact pairs have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachTrianglePointContact(IndexType f, FOnContact&& fOnContact);
    /**
     * @brief Visit all point-triangle contacts incident on triangle `f` (const).
     * @tparam FOnContact Callable with signature
     * `void(PointTriangleContactSet::ConstAccessorType C, Stencil stencil)`
     * @param f Triangle index
     * @param fOnContact Callback for each contact `(i, f)`
     * @pre Reverse contact pairs have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachTrianglePointContact(IndexType f, FOnContact&& fOnContact) const;
    /**
     * @brief Visit all edge-edge contacts for half-edge `he`.
     * @tparam FOnContact Callable with signature
     * `void(EdgeEdgeContactSet::AccessorType C, Stencil stencil)`
     * @param he Half-edge index
     * @param fOnContact Callback for each contact `(he, he')` and `(he', he)` if reverse contact
     * pairs have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachEdgeEdgeContact(IndexType he, FOnContact&& fOnContact);
    /**
     * @brief Visit all edge-edge contacts for half-edge `he` (const).
     * @tparam FOnContact Callable with signature
     * `void(EdgeEdgeContactSet::ConstAccessorType C, Stencil stencil)`
     * @param he Half-edge index
     * @param fOnContact Callback for each contact `(he, he')` and `(he', he)` if reverse contact
     * pairs have been computed via `UpdateConstraintSet(X, true)`
     */
    template <class FOnContact>
    void ForEachEdgeEdgeContact(IndexType he, FOnContact&& fOnContact) const;
    /**
     * @brief Load the stencil for a contact pair
     * @param u First mesh primitive index
     * @param v Second mesh primitive index
     * @param gu First mesh geometry index
     * @param gv Second mesh geometry index
     * @return The pair (X, nodes) of stencil (per-column) point matrix and corresponding indices
     * where `X` is a `math::linalg::mini::SMatrix<TScalar, kDims, kStencil>`
     */
    template <class TContactSet, class TDerivedx>
    auto LoadStencil(Eigen::MatrixBase<TDerivedx> const& x, Stencil const& stencil) const;
    /**
     * @brief Load the stencil point indices for a contact pair (index-only, no position loading)
     * @param stencil Contact stencil
     * @return The stencil point indices as `std::array<TIndex, kStencil>`
     */
    template <class TContactSet>
    auto LoadStencil(Stencil const& stencil) const;
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
     * @brief Compute the total potential energy from all contacts
     * @tparam TDerivedx Matrix type
     * @param x `3 x |# points|` current point positions (column-major: one point per column)
     * @return Total potential energy
     */
    template <class TDerivedx>
    ScalarType Potential(Eigen::MatrixBase<TDerivedx> const& x) const;
    /**
     * @brief Compute the total contact gradient
     * @tparam TDerivedx Matrix type
     * @param x `3 x |# points|` current point positions (column-major: one point per column)
     * @return Total contact gradient
     */
    template <class TDerivedx>
    auto Gradient(Eigen::MatrixBase<TDerivedx> const& x) const
        -> Eigen::Vector<ScalarType, Eigen::Dynamic>;
    /**
     * @brief Compute the total contact gradient and add it to `g`
     * @tparam TDerivedx Matrix type
     * @tparam TDerivedg Writeable matrix type
     * @param x `3 x |# points|` current point positions (column-major: one point per column)
     * @param g `3*|# points| x 1` or `3 x |# points|` total contact gradient
     */
    template <class TDerivedx, class TDerivedg>
    void ToGradient(Eigen::MatrixBase<TDerivedx> const& x, Eigen::MatrixBase<TDerivedg>& g) const;
    /**
     * @brief Get the total number of contacts
     * @return Total number of contacts
     */
    auto NumContacts() const
    {
        return mPointPointContacts.Size() + mPointEdgeContacts.Size() +
               mPointTriangleContacts.Size() + mEdgeEdgeContacts.Size();
    }
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
     * @brief Get the dynamic point positions
     * @return `3 x |# points|` dynamic point positions
     */
    auto DynamicPointPositions() const -> Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const&
    {
        return mXdynamic;
    }
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
     * @brief Contact pair reference sorted by `v`.
     */
    struct ReverseContactPair
    {
        IndexType v; ///< Mesh primitive index
        IndexType u; ///< Mesh primitive index
        IndexType k; ///< Contact index
    };
    /**
     * @brief Contact set view sorted by `v` for reverse contact queries
     */
    struct ReverseContactSetView
    {
        std::vector<ReverseContactPair> contacts; ///< Contact pairs sorted by `v`
        std::vector<IndexType> prefix;            ///< Prefix array for `v` in `contacts`
        /**
         * @brief Clear the reverse contact set
         */
        void Clear()
        {
            contacts.clear();
            prefix.clear();
        }
        /**
         * @brief Check if the reverse contact set is empty
         * @return true if the reverse contact set is empty, false otherwise
         */
        bool IsEmpty() const { return prefix.empty(); }
    };

    /**
     * @brief Iterate over contacts [cstart, cend) in the contact set
     * @tparam FOnContact Callable type with signature `template <class TContactSet>
     * void(typename TContactSet::AccessorType C, Stencil stencil)`
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
     * @brief Iterate over contacts [cstart, cend) in the contact set
     * @tparam FOnContact Callable type with signature `template <class TContactSet>
     * void(typename TContactSet::ConstAccessorType C, Stencil stencil)`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to iterate over
     * @param fOnContact Callback for each contact
     * @param cstart Start index for the contact set
     * @param cend End index for the contact set
     * @note Use this lower-level contact block visitor for parallel processing
     */
    template <class FOnContact, class TContactSet>
    void ForEachContact(
        TContactSet const& contactSet,
        FOnContact&& fOnContact,
        TIndex cstart,
        TIndex cend) const;
    /**
     * @brief Multi-threaded contact visitor
     * @tparam FOnContact
     * @tparam TContactSet
     * @param contactSet
     * @param fOnContact
     * @param nThreads
     * @param tg
     */
    template <class FOnContact, class TContactSet>
    void ForEachContact(
        TContactSet& contactSet,
        FOnContact&& fOnContact,
        std::int32_t nThreads,
        tbb::task_group& tg);
    /**
     * @brief Multi-threaded contact visitor
     * @tparam FOnContact
     * @tparam TContactSet
     * @param contactSet
     * @param fOnContact
     * @param nThreads
     * @param tg
     */
    template <class FOnContact, class TContactSet>
    void ForEachContact(
        TContactSet const& contactSet,
        FOnContact&& fOnContact,
        std::int32_t nThreads,
        tbb::task_group& tg) const;
    /**
     * @brief Visit contacts for source primitive `u` in a contact set.
     * @tparam FOnContact Callable with signature
     * `void(ConstraintAccessor<TContactSet> C, Stencil stencil)`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to iterate
     * @param u Source primitive index
     * @param f Callback for each contact
     */
    template <class FOnContact, class TContactSet>
    void ForEachForwardContact(TContactSet& contactSet, IndexType u, FOnContact&& f);
    /**
     * @brief Visit contacts for source primitive `u` in a contact set (const).
     * @tparam FOnContact Callable with signature
     * `void(ConstraintAccessor<TContactSet const> C, Stencil stencil)`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to iterate
     * @param u Source primitive index
     * @param f Callback for each contact
     */
    template <class FOnContact, class TContactSet>
    void ForEachForwardContact(TContactSet const& contactSet, IndexType u, FOnContact&& f) const;
    /**
     * @brief Visit contacts for target primitive `v` in a reverse contact set.
     * @tparam FOnContact Callable with signature
     * `void(ConstraintAccessor<TContactSet> C, Stencil stencil)`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to access constraint data from
     * @param reverseSet Reverse contact set view sorted by v
     * @param v Target primitive index
     * @param f Callback for each contact
     */
    template <class FOnContact, class TContactSet>
    void ForEachReverseContact(
        TContactSet& contactSet,
        ReverseContactSetView const& reverseSet,
        IndexType v,
        FOnContact&& f);
    /**
     * @brief Visit contacts for target primitive `v` in a reverse contact set (const).
     * @tparam FOnContact Callable with signature
     * `void(ConstraintAccessor<TContactSet const> C, Stencil stencil)`
     * @tparam TContactSet Contact set type
     * @param contactSet Contact set to access constraint data from
     * @param reverseSet Reverse contact set view sorted by v
     * @param v Target primitive index
     * @param f Callback for each contact
     */
    template <class FOnContact, class TContactSet>
    void ForEachReverseContact(
        TContactSet const& contactSet,
        ReverseContactSetView const& reverseSet,
        IndexType v,
        FOnContact&& f) const;
    /**
     * @brief Transfer OGC contact pairs to our contact sets
     * @param bComputeReversePairs Store reverse contact pair (j,i) for contact pair (i,j) if true
     */
    void UpdateContactSetsFromOgcPairs(bool bComputeReversePairs);
    /**
     * @brief Serialize a single constraint set to an archive group
     * @tparam TContactSet Contact set type
     * @param contactSet The contact set to serialize
     * @param grp Archive group to write into
     * @pre `contactSet` has compact IDs
     */
    template <class TContactSet>
    static void SerializeConstraintSet(TContactSet& contactSet, io::Archive& grp);
    /**
     * @brief Deserialize a single constraint set from an archive group
     * @tparam TContactSet Contact set type
     * @param contactSet The contact set to populate
     * @param grp Archive group to read from
     */
    template <class TContactSet>
    static void DeserializeConstraintSet(TContactSet& contactSet, io::Archive const& grp);
    /**
     * @brief Get the geometry prefix arrays for the contact set
     * @return The pair (prefu, prefv)
     */
    template <class TContactSet>
    auto GeometryPrefixArrays() const;
    /**
     * @brief Load a point's position from the mesh state
     * @param i Point index
     * @param g Geometry type
     * @param xi Output position vector
     * @return The point index on the corresponding geometry g
     */
    template <class TDerivedx>
    auto LoadPoint(Eigen::MatrixBase<TDerivedx> const& x, TIndex i, int g, auto&& xi) const;
    /**
     * @brief Load a half-edge's positions from the mesh state
     * @param he Half-edge index
     * @param g Geometry type
     * @param xi Output position vector for the incoming vertex
     * @param xj Output position vector for the outgoing vertex
     * @return The half-edge point indices (i, j) on the corresponding geometry g
     */
    template <class TDerivedx>
    auto LoadHalfEdge(Eigen::MatrixBase<TDerivedx> const& x, TIndex he, int g, auto&& xi, auto&& xj)
        const;
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
        Eigen::MatrixBase<TDerivedx> const& x,
        TIndex f,
        int g,
        auto&& xi,
        auto&& xj,
        auto&& xk) const;
    /**
     * @brief Load a point's index from the mesh state (index-only, no position loading)
     * @param i Point index
     * @param g Geometry type
     * @return The point index on the corresponding geometry g
     */
    auto LoadPoint(TIndex i, int g) const;
    /**
     * @brief Load a half-edge's point indices from the mesh state (index-only, no position
     * loading)
     * @param he Half-edge index
     * @param g Geometry type
     * @return The half-edge point indices (i, j) on the corresponding geometry g
     */
    auto LoadHalfEdge(TIndex he, int g) const;
    /**
     * @brief Load a triangle's point indices from the mesh state (index-only, no position loading)
     * @param f Triangle index
     * @param g Geometry type
     * @return The triangle point indices (i, j, k) on the corresponding geometry g
     */
    auto LoadTriangle(TIndex f, int g) const;
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
     * @pre `H` is symmetric
     */
    template <int kDims, int kStencil, class TDerivedH>
    TScalar RayleighQuotient(
        Eigen::SparseCompressedBase<TDerivedH> const& H,
        math::linalg::mini::SVector<TScalar, kStencil * kDims> const& gradc,
        std::array<TIndex, kStencil> const& nodes) const;

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

    ReverseContactSetView mReversePointPointContacts;    ///< Point-point reverse pairs
    ReverseContactSetView mReversePointEdgeContacts;     ///< Point-edge reverse pairs
    ReverseContactSetView mReversePointTriangleContacts; ///< Point-triangle reverse pairs
    ReverseContactSetView mReverseEdgeEdgeContacts;      ///< Edge-edge reverse pairs
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
inline MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithSequentialAugmentedLagrangian(
    TScalar gamma_,
    TScalar gammaf_,
    TScalar dmin_,
    TScalar decay_,
    TScalar decaylo_)
{
    this->gamma   = gamma_;
    this->gammaf  = gammaf_;
    this->dmin    = dmin_;
    this->decay   = decay_;
    this->decaylo = decaylo_;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::Construct(bool bValidate)
{
    mOgcParams.Construct(bValidate);
    if (bValidate)
    {
        if (kc < TScalar(0))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): kc must be non-negative.");
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
        if (gamma <= TScalar(0))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): gamma must be positive.");
        }
        if (dmin <= TScalar(0))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): dmin must be positive.");
        }
        if (dmin >= mOgcParams.r)
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): dmin must satisfy dmin < r.");
        }
        if (decay < TScalar(0) or decay > TScalar(1))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): decay must satisfy 0 <= decay <= 1.");
        }
        if (decaylo < TScalar(0) or decaylo > TScalar(1))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): decaylo must satisfy 0 <= decaylo <= 1.");
        }
    }
    UpdateOgcTwoStageActivationCoefficients();
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::UpdateOgcTwoStageActivationCoefficients()
{
    auto tau = TScalar(0.5) * mOgcParams.r;
    auto r   = mOgcParams.r;
    kcp      = tau * kc * (tau - r) * (tau - r);
    b        = (TScalar(0.5) * kc) * (r - tau) * (r - tau) + kcp * std::log(tau);
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
    grp.WriteMetaData("gamma", gamma);
    grp.WriteMetaData("gammaf", gammaf);
    grp.WriteMetaData("dmin", dmin);
    grp.WriteMetaData("decay", decay);
    grp.WriteMetaData("decaylo", decaylo);
    grp.WriteMetaData("bDeactivate", static_cast<int>(bDeactivate));
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
    if (grp.HasMetaData("gamma"))
        gamma = grp.ReadMetaData<TScalar>("gamma");
    if (grp.HasMetaData("gammaf"))
        gammaf = grp.ReadMetaData<TScalar>("gammaf");
    if (grp.HasMetaData("dmin"))
        dmin = grp.ReadMetaData<TScalar>("dmin");
    if (grp.HasMetaData("decay"))
        decay = grp.ReadMetaData<TScalar>("decay");
    if (grp.HasMetaData("decaylo"))
        decaylo = grp.ReadMetaData<TScalar>("decaylo");
    if (grp.HasMetaData("bDeactivate"))
        bDeactivate = static_cast<bool>(grp.ReadMetaData<int>("bDeactivate"));
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
    mPointPointContacts.Clear();
    mPointEdgeContacts.Clear();
    mPointTriangleContacts.Clear();
    mEdgeEdgeContacts.Clear();
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
inline void MeshDynamics<TScalar, TIndex>::UpdateConstraintSet(
    Eigen::DenseBase<TDerivedX> const& X,
    bool bComputeReverseContactPairs)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.UpdateConstraintSet");
    if (mParams.bDeactivate)
    {
        mOgcState.bv.setConstant(std::numeric_limits<ScalarType>::max());
        mPointPointContacts.Clear();
        mPointEdgeContacts.Clear();
        mPointTriangleContacts.Clear();
        mEdgeEdgeContacts.Clear();
        mReversePointPointContacts.Clear();
        mReversePointEdgeContacts.Clear();
        mReversePointTriangleContacts.Clear();
        mReverseEdgeEdgeContacts.Clear();
        mRequiresBoundsRecomputation = false;
        mNumTruncatedPoints          = 0;
        return;
    }
    mXdynamic = X.derived();
    mOgcState.PrepareForExecution(mOgcInput, mParams.mOgcParams);
    ogc::Execute(mOgcInput, mParams.mOgcParams, mOgcState);
    UpdateContactSetsFromOgcPairs(bComputeReverseContactPairs);
    mRequiresBoundsRecomputation = false;
    mNumTruncatedPoints          = 0;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx, class TDerivedxt>
inline void MeshDynamics<TScalar, TIndex>::LinearizeConstraints(
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedxt> const& xt)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.LinearizeConstraints");
    auto const nThreads = static_cast<std::int32_t>(std::thread::hardware_concurrency());
    ForAllContacts(
        [&]<class TContactSet>(
            typename TContactSet::AccessorType C,
            Stencil stencil,
            std::int32_t /*t*/) {
            using ConstraintAccessorType   = decltype(C);
            static auto constexpr kDims    = ConstraintAccessorType::kDims;
            static auto constexpr kStencil = ConstraintAccessorType::kStencil;
            auto const [Xc, _n1]           = LoadStencil<TContactSet>(x, stencil);
            auto const [Xtc, _n2]          = LoadStencil<TContactSet>(xt, stencil);
            auto xc                        = Reshape<ConstraintAccessorType::kDofs, 1>(Xc);
            auto xtc                       = Reshape<ConstraintAccessorType::kDofs, 1>(Xtc);
            C.ComputeLinearization(xc);
            auto F = C.Friction();
            F.ComputeLinearization(xc, xtc);
        },
        nThreads);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerived>
inline void MeshDynamics<TScalar, TIndex>::UpdatePenaltyParameter(
    Eigen::SparseCompressedBase<TDerived> const& H)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.UpdatePenaltyParameter");
    auto nThreads = static_cast<std::int32_t>(std::thread::hardware_concurrency());
    mParams.kc    = 0;
    ForAllContacts(
        [&]<class TContactSet>(
            typename TContactSet::AccessorType C,
            Stencil stencil,
            std::int32_t /*t*/) {
            using ConstraintAccessorType   = decltype(C);
            static auto constexpr kDims    = ConstraintAccessorType::kDims;
            static auto constexpr kStencil = ConstraintAccessorType::kStencil;
            auto nodes                     = LoadStencil<TContactSet>(stencil);
            TScalar Q                      = RayleighQuotient<kDims, kStencil>(H, C.Grad(), nodes);
            common::AtomicMin(mParams.kc, -Q);
        },
        nThreads);
    mParams.kc = -mParams.kc;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <int Mask, class TDerivedx>
inline void MeshDynamics<TScalar, TIndex>::UpdateDual(Eigen::MatrixBase<TDerivedx> const& x)
{
    auto nThreads = static_cast<std::int32_t>(std::thread::hardware_concurrency());
    ForAllContacts(
        [&]<class TContactSet>(
            typename TContactSet::AccessorType C,
            Stencil stencil,
            std::int32_t /*t*/) {
            using ConstraintAccessorType = decltype(C);
            auto const [Xc, nodes]       = LoadStencil<TContactSet>(x, stencil);
            auto xc                      = Reshape<ConstraintAccessorType::kDofs, 1>(Xc);
            C.Eval()                     = C.Eval(xc);
            auto F                       = C.Friction();
            F.Eval()                     = F.Eval(xc);
            auto kn                      = mParams.gamma * mParams.kc;
            if (static_cast<bool>(Mask & EDualVariable::Slack))
                C.Slack() = std::max(TScalar(0), C.Eval() - mParams.dmin - C.Lambda() / kn);
            if (static_cast<bool>(Mask & EDualVariable::LagrangeMultiplier))
            {
                if (C.Slack() == TScalar(0))
                {
                    C.Lambda() -= kn * (C.Eval() - mParams.dmin);
                    C.Decay() = TScalar(1);
                    auto muS  = mParams.mu;
                    auto kf   = mParams.gammaf * mParams.kc;
                    F.Lambda() -= kf * F.Eval();
                    TScalar frictionLimit = muS * C.Lambda();
                    TScalar lambdafn2     = SquaredNorm(F.Lambda());
                    if (lambdafn2 > frictionLimit * frictionLimit)
                    {
                        using std::sqrt;
                        F.Lambda() *= frictionLimit / sqrt(lambdafn2);
                    }
                }
                else
                {
                    C.Lambda() = TScalar(0);
                    F.Lambda().SetZero();
                    C.Decay() *= mParams.decay;
                }
            }
        },
        nThreads);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForAllContacts(FOnContact&& fOnContact, std::int32_t nThreads)
{
    tbb::global_control gc{
        tbb::global_control::max_allowed_parallelism,
        static_cast<std::size_t>(nThreads)};
    tbb::task_group tg;
    ForEachContact(mPointPointContacts, fOnContact, nThreads, tg);
    ForEachContact(mPointEdgeContacts, fOnContact, nThreads, tg);
    ForEachContact(mPointTriangleContacts, fOnContact, nThreads, tg);
    ForEachContact(mEdgeEdgeContacts, fOnContact, nThreads, tg);
    tg.wait();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForAllContacts(FOnContact&& fOnContact, std::int32_t nThreads) const
{
    tbb::global_control gc{
        tbb::global_control::max_allowed_parallelism,
        static_cast<std::size_t>(nThreads)};
    tbb::task_group tg;
    ForEachContact(mPointPointContacts, fOnContact, nThreads, tg);
    ForEachContact(mPointEdgeContacts, fOnContact, nThreads, tg);
    ForEachContact(mPointTriangleContacts, fOnContact, nThreads, tg);
    ForEachContact(mEdgeEdgeContacts, fOnContact, nThreads, tg);
    tg.wait();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachContact(
    TContactSet& contactSet,
    FOnContact&& fOnContact,
    std::int32_t nThreads)
{
    tbb::task_group tg;
    ForEachContact(contactSet, std::forward<FOnContact>(fOnContact), nThreads, tg);
    tg.wait();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachContact(
    TContactSet const& contactSet,
    FOnContact&& fOnContact,
    std::int32_t nThreads) const
{
    tbb::task_group tg;
    ForEachContact(contactSet, std::forward<FOnContact>(fOnContact), nThreads, tg);
    tg.wait();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachPointPointContact(IndexType i, FOnContact&& fOnContact)
{
    ForEachForwardContact(mPointPointContacts, i, fOnContact);
    ForEachReverseContact(mPointPointContacts, mReversePointPointContacts, i, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachPointPointContact(IndexType i, FOnContact&& fOnContact) const
{
    ForEachForwardContact(mPointPointContacts, i, fOnContact);
    ForEachReverseContact(mPointPointContacts, mReversePointPointContacts, i, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachPointEdgeContact(IndexType i, FOnContact&& fOnContact)
{
    ForEachForwardContact(mPointEdgeContacts, i, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachPointEdgeContact(IndexType i, FOnContact&& fOnContact) const
{
    ForEachForwardContact(mPointEdgeContacts, i, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachEdgePointContact(IndexType he, FOnContact&& fOnContact)
{
    ForEachReverseContact(mPointEdgeContacts, mReversePointEdgeContacts, he, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachEdgePointContact(IndexType he, FOnContact&& fOnContact) const
{
    ForEachReverseContact(mPointEdgeContacts, mReversePointEdgeContacts, he, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachPointTriangleContact(IndexType i, FOnContact&& fOnContact)
{
    ForEachForwardContact(mPointTriangleContacts, i, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachPointTriangleContact(
    IndexType i,
    FOnContact&& fOnContact) const
{
    ForEachForwardContact(mPointTriangleContacts, i, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachTrianglePointContact(IndexType f, FOnContact&& fOnContact)
{
    ForEachReverseContact(mPointTriangleContacts, mReversePointTriangleContacts, f, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachTrianglePointContact(
    IndexType f,
    FOnContact&& fOnContact) const
{
    ForEachReverseContact(mPointTriangleContacts, mReversePointTriangleContacts, f, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachEdgeEdgeContact(IndexType he, FOnContact&& fOnContact)
{
    ForEachForwardContact(mEdgeEdgeContacts, he, fOnContact);
    ForEachReverseContact(mEdgeEdgeContacts, mReverseEdgeEdgeContacts, he, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact>
inline void
MeshDynamics<TScalar, TIndex>::ForEachEdgeEdgeContact(IndexType he, FOnContact&& fOnContact) const
{
    ForEachForwardContact(mEdgeEdgeContacts, he, fOnContact);
    ForEachReverseContact(mEdgeEdgeContacts, mReverseEdgeEdgeContacts, he, fOnContact);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TContactSet, class TDerivedx>
inline auto MeshDynamics<TScalar, TIndex>::LoadStencil(
    Eigen::MatrixBase<TDerivedx> const& x,
    Stencil const& stencil) const
{
    using ConstraintAccessorType   = typename TContactSet::ConstAccessorType;
    static auto constexpr kDims    = ConstraintAccessorType::kDims;
    static auto constexpr kStencil = ConstraintAccessorType::kStencil;
    using math::linalg::mini::SMatrix;
    SMatrix<TScalar, kDims, kStencil> X;
    std::array<TIndex, kStencil> nodes;
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
inline auto MeshDynamics<TScalar, TIndex>::LoadStencil(Stencil const& stencil) const
{
    using ContactSetType         = std::remove_cvref_t<TContactSet>;
    using ConstraintAccessorType = typename ContactSetType::ConstAccessorType;
    std::array<TIndex, ConstraintAccessorType::kStencil> nodes;
    if constexpr (std::is_same_v<ContactSetType, PointPointContactSet>)
    {
        nodes[0] = LoadPoint(stencil.u, stencil.gu);
        nodes[1] = LoadPoint(stencil.v, stencil.gv);
    }
    else if constexpr (std::is_same_v<ContactSetType, PointEdgeContactSet>)
    {
        nodes[0]                     = LoadPoint(stencil.u, stencil.gu);
        std::tie(nodes[1], nodes[2]) = LoadHalfEdge(stencil.v, stencil.gv);
    }
    else if constexpr (std::is_same_v<ContactSetType, PointTriangleContactSet>)
    {
        nodes[0]                               = LoadPoint(stencil.u, stencil.gu);
        std::tie(nodes[1], nodes[2], nodes[3]) = LoadTriangle(stencil.v, stencil.gv);
    }
    else if constexpr (std::is_same_v<ContactSetType, EdgeEdgeContactSet>)
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
template <class TDerivedx>
inline TScalar MeshDynamics<TScalar, TIndex>::Potential(Eigen::MatrixBase<TDerivedx> const& x) const
{
    TScalar E{0};
    ForAllContacts(
        [&]<class TContactSet>(
            typename TContactSet::ConstAccessorType C,
            Stencil stencil,
            std::int32_t /*t*/) {
            using ConstraintAccessorType   = decltype(C);
            static auto constexpr kStencil = ConstraintAccessorType::kStencil;
            static auto constexpr kDims    = ConstraintAccessorType::kDims;
            static auto constexpr kDofs    = ConstraintAccessorType::kDofs;
            static_assert(kDims == 3, "Only 3D is supported");
            auto const [Xc, nodes] = LoadStencil<TContactSet>(x, stencil);
            auto xc                = Reshape<kDofs, 1>(Xc);
            TScalar cs             = C.Eval(xc) - mParams.dmin - C.Slack();
            auto kn                = mParams.gamma * mParams.kc;
            E += /*C.Decay() **/ (TScalar(0.5) * kn * cs * cs - C.Lambda() * cs);
            auto F  = C.Friction();
            auto cf = F.Eval(xc);
            auto kf = mParams.gammaf * mParams.kc;
            E += TScalar(0.5) * kf * Dot(cf, cf) - Dot(F.Lambda(), cf);
        },
        1 /*nThreads*/);
    return E;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx>
inline auto MeshDynamics<TScalar, TIndex>::Gradient(Eigen::MatrixBase<TDerivedx> const& x) const
    -> Eigen::Vector<TScalar, Eigen::Dynamic>
{
    Eigen::Vector<TScalar, Eigen::Dynamic> grad(mXdynamic.size());
    grad.setZero();
    ToGradient(x, grad);
    return grad;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx, class TDerivedg>
inline void MeshDynamics<TScalar, TIndex>::ToGradient(
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::MatrixBase<TDerivedg>& g_) const
{
    auto g                    = g_.derived().reshaped();
    Eigen::Index const nNodes = g.size() / 3;
    ForAllContacts(
        [&]<class TContactSet>(
            typename TContactSet::ConstAccessorType C,
            Stencil stencil,
            std::int32_t /*t*/) {
            static auto constexpr kStencil = C.kStencil;
            static auto constexpr kDims    = C.kDims;
            static auto constexpr kDofs    = C.kDofs;
            static_assert(kDims == 3, "Only 3D is supported");
            auto const [XC, nodes] = LoadStencil<TContactSet>(x, stencil);
            auto xc                = Reshape<kDofs, 1>(XC);
            auto kn                = mParams.gamma * mParams.kc;
            TScalar cs             = C.Eval(xc) - mParams.dmin - C.Slack();
            TScalar dL             = /*C.Decay() **/ (kn * cs - C.Lambda());
            // Friction gradient \nabla_x [ 0.5*kf*||c_f||^2 - lambda_f^T c_f ]
            // = [ kf * c_f - lambda_f ] \nabla_x c_f where \nabla_xi c_f is W(i)*T
            auto F        = C.Friction();
            auto const& W = F.Weights();
            auto const& T = F.TangentBasis();
            auto cf       = F.Eval(xc);
            auto kf       = mParams.gammaf * mParams.kc;
            using math::linalg::mini::SVector;
            SVector<TScalar, 2> df     = kf * cf - F.Lambda();
            SVector<TScalar, kDims> gf = T * df;
            using math::linalg::mini::ToEigen;
            auto gradnc = ToEigen(C.Grad());
            auto gradfc = ToEigen(gf);
            for (auto ki = 0; ki < kStencil; ++ki)
            {
                if (nodes[ki] < nNodes)
                {
                    g.template segment<kDims>(nodes[ki] * kDims) +=
                        dL * gradnc.template segment<kDims>(ki * kDims) + W(ki) * gradfc;
                }
            }
        },
        1 /*nThreads*/);
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
    {
        auto pointPointContactsGrp = grp["mPointPointContacts"];
        SerializeConstraintSet(mPointPointContacts, pointPointContactsGrp);
    }
    {
        auto pointEdgeContactsGrp = grp["mPointEdgeContacts"];
        SerializeConstraintSet(mPointEdgeContacts, pointEdgeContactsGrp);
    }
    {
        auto pointTriangleContactsGrp = grp["mPointTriangleContacts"];
        SerializeConstraintSet(mPointTriangleContacts, pointTriangleContactsGrp);
    }
    {
        auto edgeEdgeContactsGrp = grp["mEdgeEdgeContacts"];
        SerializeConstraintSet(mEdgeEdgeContacts, edgeEdgeContactsGrp);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Deserialize(io::Archive const& archive)
{
    auto grp = archive["pbat.sim.contact.MeshDynamics"];
    mParams.Deserialize(grp["mParams"]);
    if (grp.HasMetaData("mNumTruncatedPoints"))
    {
        mNumTruncatedPoints = grp.ReadMetaData<Eigen::Index>("mNumTruncatedPoints");
    }
    if (grp.HasMetaData("mRequiresBoundsRecomputation"))
    {
        mRequiresBoundsRecomputation =
            static_cast<bool>(grp.ReadMetaData<int>("mRequiresBoundsRecomputation"));
    }
    if (grp.HasData("mXdynamic") and grp.HasGroup("mDynamicMeshes"))
    {
        mXdynamic =
            grp.ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("mXdynamic");
        mDynamicMeshes.Deserialize(grp["mDynamicMeshes"]);
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
    if (grp.HasGroup("mPointPointContacts"))
    {
        DeserializeConstraintSet(mPointPointContacts, grp["mPointPointContacts"]);
    }
    if (grp.HasGroup("mPointEdgeContacts"))
    {
        DeserializeConstraintSet(mPointEdgeContacts, grp["mPointEdgeContacts"]);
    }
    if (grp.HasGroup("mPointTriangleContacts"))
    {
        DeserializeConstraintSet(mPointTriangleContacts, grp["mPointTriangleContacts"]);
    }
    if (grp.HasGroup("mEdgeEdgeContacts"))
    {
        DeserializeConstraintSet(mEdgeEdgeContacts, grp["mEdgeEdgeContacts"]);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TContactSet>
void MeshDynamics<TScalar, TIndex>::SerializeConstraintSet(
    TContactSet& contactSet,
    io::Archive& grp)
{
    static constexpr int kDofs    = TContactSet::AccessorType::kDofs;
    static constexpr int kStencil = TContactSet::AccessorType::kStencil;
    // NOTE: The contact set should already be compacted.
    // contactSet.CompactIds();
    auto const nAdj         = static_cast<Eigen::Index>(contactSet.Size());
    auto const& adjacencies = contactSet.Adjacencies();
    auto const& prefix      = contactSet.Prefix();
    // Adjacencies as 3 x nAdj matrix (source, target, id)
    Eigen::Matrix<TIndex, 3, Eigen::Dynamic> adj(3, nAdj);
    for (Eigen::Index i = 0; i < nAdj; ++i)
    {
        adj(0, i) = std::get<0>(adjacencies[i]);
        adj(1, i) = std::get<1>(adjacencies[i]);
        adj(2, i) = std::get<2>(adjacencies[i]);
    }
    grp.WriteData("adjacencies", adj);
    grp.WriteData("prefix", prefix);
    // Constraint data
    auto const& lambdas = contactSet.template Data<0>();
    auto const& slacks  = contactSet.template Data<1>();
    auto const& decays  = contactSet.template Data<2>();
    auto const& chats   = contactSet.template Data<3>();
    auto const& gradcs  = contactSet.template Data<4>();
    auto const& evals   = contactSet.template Data<5>();
    grp.WriteData("lambda", lambdas);
    grp.WriteData("slack", slacks);
    grp.WriteData("chat", chats);
    grp.WriteData("eval", evals);
    // Decay (extract gamma)
    {
        std::vector<TScalar> decayVec(nAdj);
        for (Eigen::Index i = 0; i < nAdj; ++i)
            decayVec[i] = decays[i].gamma;
        grp.WriteData("decay", decayVec);
    }
    // Gradient as kDofs x nAdj matrix
    {
        Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> gradcsEig(kDofs, nAdj);
        for (Eigen::Index i = 0; i < nAdj; ++i)
        {
            auto const& gradc = gradcs[i];
            gradcsEig.col(i)  = ToEigen(gradc);
        }
        grp.WriteData("gradc", gradcsEig);
    }
    // Friction data
    auto const& tangentBases   = contactSet.template Data<6>();
    auto const& tangentWeights = contactSet.template Data<7>();
    auto const& cf             = contactSet.template Data<8>();
    auto const& dhat           = contactSet.template Data<9>();
    auto const& lambdaf        = contactSet.template Data<10>();
    using math::linalg::mini::ToEigen;
    // Tangent basis as 3 x 2*nAdj matrix
    {
        Eigen::Matrix<TScalar, 3, Eigen::Dynamic> Teig(3, 2 * nAdj);
        for (Eigen::Index i = 0; i < nAdj; ++i)
        {
            auto const& Ti                      = tangentBases[i];
            Teig.template block<3, 2>(0, 2 * i) = ToEigen(Ti);
        }
        grp.WriteData("tangentBasis", Teig);
    }
    // Weights as kStencil x nAdj matrix
    {
        Eigen::Matrix<TScalar, kStencil, Eigen::Dynamic> Weig(kStencil, nAdj);
        for (Eigen::Index i = 0; i < nAdj; ++i)
        {
            auto const& Wi = tangentWeights[i];
            Weig.col(i)    = ToEigen(Wi);
        }
        grp.WriteData("tangentWeights", Weig);
    }
    // Friction constraint value c_f(x) as 2 x nAdj matrix
    {
        Eigen::Matrix<TScalar, 2, Eigen::Dynamic> cfEig(2, nAdj);
        for (Eigen::Index i = 0; i < nAdj; ++i)
            cfEig.col(i) = ToEigen(cf[i]);
        grp.WriteData("cf", cfEig);
    }
    // dhat = T^T W xt as 2 x nAdj matrix
    {
        Eigen::Matrix<TScalar, 2, Eigen::Dynamic> dhatEig(2, nAdj);
        for (Eigen::Index i = 0; i < nAdj; ++i)
            dhatEig.col(i) = ToEigen(dhat[i]);
        grp.WriteData("dhat", dhatEig);
    }
    // Friction Lagrange multiplier lambda_f as 2 x nAdj matrix
    {
        Eigen::Matrix<TScalar, 2, Eigen::Dynamic> lfEig(2, nAdj);
        for (Eigen::Index i = 0; i < nAdj; ++i)
            lfEig.col(i) = ToEigen(lambdaf[i]);
        grp.WriteData("lambdaf", lfEig);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TContactSet>
void MeshDynamics<TScalar, TIndex>::DeserializeConstraintSet(
    TContactSet& contactSet,
    io::Archive const& grp)
{
    static constexpr int kDofs    = TContactSet::AccessorType::kDofs;
    static constexpr int kStencil = TContactSet::AccessorType::kStencil;
    // Read adjacencies + prefix
    auto adj = grp.ReadData<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>>("adjacencies");
    auto const nAdj = adj.cols();
    std::vector<std::tuple<TIndex, TIndex, TIndex>> adjacencies(nAdj);
    for (Eigen::Index i = 0; i < nAdj; ++i)
        adjacencies[i] = {adj(0, i), adj(1, i), adj(2, i)};
    std::vector<TIndex> prefix = grp.ReadData<std::vector<TIndex>>("prefix");
    // Read scalar data
    std::vector<TScalar> lambdas = grp.ReadData<std::vector<TScalar>>("lambda");
    std::vector<TScalar> slacks  = grp.ReadData<std::vector<TScalar>>("slack");
    std::vector<TScalar> chats   = grp.ReadData<std::vector<TScalar>>("chat");
    std::vector<TScalar> evals   = grp.ReadData<std::vector<TScalar>>("eval");
    std::vector<TScalar> decaysS = grp.ReadData<std::vector<TScalar>>("decay");
    std::vector<math::linalg::mini::SVector<TScalar, kDofs>> gradcs(nAdj);
    auto gradcsEig = grp.ReadData<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>>("gradc");
    // Build data vectors
    std::vector<Decay> decays(nAdj);
    for (Eigen::Index i = 0; i < nAdj; ++i)
    {
        decays[i].gamma = decaysS[i];
        auto& gradc     = gradcs[i];
        for (int d = 0; d < kDofs; ++d)
            gradc(d) = gradcsEig(d, i);
    }
    // Friction data (optional for backward compatibility)
    using SMatrix32         = math::linalg::mini::SMatrix<TScalar, 3, 2>;
    using SVector2          = math::linalg::mini::SVector<TScalar, 2>;
    using WeightsVectorType = math::linalg::mini::SVector<TScalar, kStencil>;
    std::vector<SMatrix32> tangentBases(nAdj);
    std::vector<WeightsVectorType> tangentWeights(nAdj);
    std::vector<SVector2> cf(nAdj);
    std::vector<SVector2> dhat(nAdj);
    std::vector<SVector2> lambdaf(nAdj);
    using math::linalg::mini::FromEigen;
    if (grp.HasData("tangentBasis"))
    {
        auto tbEig =
            grp.ReadData<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>>("tangentBasis");
        for (Eigen::Index i = 0; i < nAdj; ++i)
            tangentBases[i] = FromEigen(tbEig.template block<3, 2>(0, 2 * i));
    }
    if (grp.HasData("tangentWeights"))
    {
        auto Weig =
            grp.ReadData<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>>("tangentWeights");
        for (Eigen::Index i = 0; i < nAdj; ++i)
            tangentWeights[i] = FromEigen(Weig.col(i).template head<kStencil>());
    }
    if (grp.HasData("cf"))
    {
        auto cfEig = grp.ReadData<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>>("cf");
        for (Eigen::Index i = 0; i < nAdj; ++i)
            cf[i] = FromEigen(cfEig.col(i).template head<2>());
    }
    if (grp.HasData("dhat"))
    {
        auto dhatEig = grp.ReadData<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>>("dhat");
        for (Eigen::Index i = 0; i < nAdj; ++i)
            dhat[i] = FromEigen(dhatEig.col(i).template head<2>());
    }
    if (grp.HasData("lambdaf"))
    {
        auto lfEig =
            grp.ReadData<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>>("lambdaf");
        for (Eigen::Index i = 0; i < nAdj; ++i)
            lambdaf[i] = FromEigen(lfEig.col(i).template head<2>());
    }
    // Construct the contact set from compact state
    contactSet.Construct(
        std::move(adjacencies),
        std::move(prefix),
        std::make_tuple(
            std::move(lambdas),
            std::move(slacks),
            std::move(decays),
            std::move(chats),
            std::move(gradcs),
            std::move(evals),
            std::move(tangentBases),
            std::move(tangentWeights),
            std::move(cf),
            std::move(dhat),
            std::move(lambdaf)));
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
        ConstraintAccessor<TContactSet> C{set, k};
        fOnContact.template operator()<TContactSet>(C, Stencil{u, v, gu, gv});
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachContact(
    TContactSet const& set,
    FOnContact&& fOnContact,
    TIndex cstart,
    TIndex cend) const
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
        ConstraintAccessor<TContactSet const> C{set, k};
        fOnContact.template operator()<TContactSet>(C, Stencil{u, v, gu, gv});
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachContact(
    TContactSet& contactSet,
    FOnContact&& fOnContact,
    std::int32_t nThreads,
    tbb::task_group& tg)
{
    tbb::global_control gc{
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(nThreads)};
    auto const fForEachThread = [&tg, nThreads](auto&& f) {
        for (std::int32_t t = 0; t < nThreads; ++t)
            tg.run([f, t]() { f(t); });
    };
    fForEachThread([&](std::int32_t t) {
        TIndex const nConstraints          = static_cast<TIndex>(contactSet.Size());
        TIndex const nConstraintsPerThread = (nConstraints + nThreads - 1) / nThreads;
        TIndex const cstart                = t * nConstraintsPerThread;
        TIndex const cend = std::min((t + 1) * nConstraintsPerThread, nConstraints);
        auto const fWrap =
            [&fOnContact,
             t = t]<class TContactSet>(typename TContactSet::AccessorType C, Stencil stencil) {
                fOnContact.template operator()<TContactSet>(C, std::move(stencil), t);
            };
        ForEachContact(contactSet, fWrap, cstart, cend);
    });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachContact(
    TContactSet const& contactSet,
    FOnContact&& fOnContact,
    std::int32_t nThreads,
    tbb::task_group& tg) const
{
    tbb::global_control gc{
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(nThreads)};
    auto const fForEachThread = [&tg, nThreads](auto&& f) {
        for (std::int32_t t = 0; t < nThreads; ++t)
            tg.run([f, t]() { f(t); });
    };
    fForEachThread([&](std::int32_t t) {
        TIndex const nConstraints          = static_cast<TIndex>(contactSet.Size());
        TIndex const nConstraintsPerThread = (nConstraints + nThreads - 1) / nThreads;
        TIndex const cstart                = t * nConstraintsPerThread;
        TIndex const cend = std::min((t + 1) * nConstraintsPerThread, nConstraints);
        auto const fWrap =
            [&fOnContact,
             t = t]<class TContactSet>(typename TContactSet::ConstAccessorType C, Stencil stencil) {
                fOnContact.template operator()<TContactSet>(C, std::move(stencil), t);
            };
        ForEachContact(contactSet, fWrap, cstart, cend);
    });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachForwardContact(
    TContactSet& contactSet,
    IndexType u,
    FOnContact&& f)
{
    auto const [prefixu, prefixv] = GeometryPrefixArrays<TContactSet>();
    int gu{0};
    while (u >= prefixu[gu + 1])
        ++gu;
    int gv{0};
    contactSet.ForEach(u, [&](auto cu, auto v, auto k) {
        while (v >= prefixv[gv + 1])
            ++gv;
        ConstraintAccessor<TContactSet const> C{contactSet, k};
        f(C, Stencil{u, v, gu, gv});
    });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachForwardContact(
    TContactSet const& contactSet,
    IndexType u,
    FOnContact&& f) const
{
    auto const [prefixu, prefixv] = GeometryPrefixArrays<TContactSet>();
    int gu{0};
    while (u >= prefixu[gu + 1])
        ++gu;
    int gv{0};
    contactSet.ForEach(u, [&](auto cu, auto v, auto k) {
        while (v >= prefixv[gv + 1])
            ++gv;
        ConstraintAccessor<TContactSet const> C{contactSet, k};
        f(C, Stencil{u, v, gu, gv});
    });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachReverseContact(
    TContactSet& contactSet,
    ReverseContactSetView const& reverseSet,
    IndexType v,
    FOnContact&& f)
{
    if (reverseSet.IsEmpty() or v >= static_cast<IndexType>(reverseSet.prefix.size() - 1))
        return;
    auto const [prefixu, prefixv] = GeometryPrefixArrays<TContactSet>();
    int gv{0};
    while (v >= prefixv[gv + 1])
        ++gv;
    int gu{0};
    auto cpbegin = reverseSet.prefix[v];
    auto cpend   = reverseSet.prefix[v + 1];
    for (TIndex cp = cpbegin; cp < cpend; ++cp)
    {
        auto const& rcp = reverseSet.contacts[cp];
        while (rcp.u >= prefixu[gu + 1])
            ++gu;
        ConstraintAccessor<TContactSet> C{contactSet, rcp.k};
        f(C, Stencil{rcp.u, v, gu, gv});
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnContact, class TContactSet>
inline void MeshDynamics<TScalar, TIndex>::ForEachReverseContact(
    TContactSet const& contactSet,
    ReverseContactSetView const& reverseSet,
    IndexType v,
    FOnContact&& f) const
{
    if (reverseSet.IsEmpty() or v >= static_cast<IndexType>(reverseSet.prefix.size() - 1))
        return;
    auto const [prefixu, prefixv] = GeometryPrefixArrays<TContactSet>();
    int gv{0};
    while (v >= prefixv[gv + 1])
        ++gv;
    int gu{0};
    auto cpbegin = reverseSet.prefix[v];
    auto cpend   = reverseSet.prefix[v + 1];
    for (TIndex cp = cpbegin; cp < cpend; ++cp)
    {
        auto const& rcp = reverseSet.contacts[cp];
        while (rcp.u >= prefixu[gu + 1])
            ++gu;
        ConstraintAccessor<TContactSet const> C{contactSet, rcp.k};
        f(C, Stencil{rcp.u, v, gu, gv});
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::UpdateContactSetsFromOgcPairs(bool bComputeReversePairs)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.UpdateContactSetsFromOgcPairs");
    using OgcStateType = decltype(mOgcState);
    tbb::task_group tg;
    auto const fUpdateContactSet = [&](auto& set, auto& newSet, auto nSourcePrimitives) {
        // set.Union(newSet);
        // set.RemoveIf([&]([[maybe_unused]] TIndex u, [[maybe_unused]] TIndex v, TIndex k) {
        //     using ContactSetType = std::remove_cvref_t<decltype(set)>;
        //     ConstraintAccessor<ContactSetType> C{set, k};
        //     return C.Decay() < mParams.decaylo;
        // });
        set.Assign(newSet);
        set.Finalize(nSourcePrimitives);
        set.CompactIds();
    };
    auto const fBuildReversePairs =
        [](auto const& set, auto nTargetNodes, ReverseContactSetView& reverseSet) {
            auto const n = static_cast<TIndex>(set.Size());
            reverseSet.contacts.resize(n);
            reverseSet.prefix.resize(nTargetNodes + 1);
            std::fill(reverseSet.prefix.begin(), reverseSet.prefix.end(), 0);
            for (TIndex c = 0; c < n; ++c)
            {
                auto const [u, v, k]   = set.WeightedAdjacency(c);
                reverseSet.contacts[c] = ReverseContactPair{v, u, k};
                ++reverseSet.prefix[v];
            }
            tbb::parallel_sort(
                reverseSet.contacts.begin(),
                reverseSet.contacts.end(),
                [](ReverseContactPair const& a, ReverseContactPair const& b) {
                    // Technically, we only need to sort by v, but also sorting by u and k generally
                    // helps with cache locality.
                    return std::tie(a.v, a.u, a.k) < std::tie(b.v, b.u, b.k);
                });
            std::exclusive_scan(
                reverseSet.prefix.begin(),
                reverseSet.prefix.end(),
                reverseSet.prefix.begin(),
                TIndex{0});
        };
    auto const nPoints    = mOgcState.mPointGeometryPrefix[OgcStateType::EGeometry::Count];
    auto const nHalfEdges = mOgcState.mHalfEdgeGeometryPrefix[OgcStateType::EGeometry::Count];
    auto const nTriangles = mOgcState.mTriangleGeometryPrefix[OgcStateType::EGeometry::Count];
    tg.run([&] { fUpdateContactSet(mPointPointContacts, mOgcState.mXX, nPoints); });
    tg.run([&] { fUpdateContactSet(mPointEdgeContacts, mOgcState.mXE, nPoints); });
    tg.run([&] { fUpdateContactSet(mPointTriangleContacts, mOgcState.mXF, nPoints); });
    tg.run([&] { fUpdateContactSet(mEdgeEdgeContacts, mOgcState.mEE, nHalfEdges); });
    tg.wait();
    if (bComputeReversePairs)
    {
        // tg.run(
        //     [&] { fBuildReversePairs(mPointPointContacts, nPoints, mReversePointPointContacts);
        //     });
        // tg.run(
        //     [&] { fBuildReversePairs(mPointEdgeContacts, nHalfEdges, mReversePointEdgeContacts);
        //     });
        // tg.run([&] {
        //     fBuildReversePairs(mPointTriangleContacts, nTriangles,
        //     mReversePointTriangleContacts);
        // });
        // tg.run(
        //     [&] { fBuildReversePairs(mEdgeEdgeContacts, nHalfEdges, mReverseEdgeEdgeContacts);
        //     });
        // tg.wait();
        fBuildReversePairs(mPointPointContacts, nPoints, mReversePointPointContacts);
        fBuildReversePairs(mPointEdgeContacts, nHalfEdges, mReversePointEdgeContacts);
        fBuildReversePairs(mPointTriangleContacts, nTriangles, mReversePointTriangleContacts);
        fBuildReversePairs(mEdgeEdgeContacts, nHalfEdges, mReverseEdgeEdgeContacts);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TContactSet>
inline auto MeshDynamics<TScalar, TIndex>::GeometryPrefixArrays() const
{
    using ContactSetType = std::remove_cvref_t<TContactSet>;
    if constexpr (std::is_same_v<ContactSetType, PointPointContactSet>)
    {
        return std::make_pair(mOgcState.mPointGeometryPrefix, mOgcState.mPointGeometryPrefix);
    }
    else if constexpr (std::is_same_v<ContactSetType, PointEdgeContactSet>)
    {
        return std::make_pair(mOgcState.mPointGeometryPrefix, mOgcState.mHalfEdgeGeometryPrefix);
    }
    else if constexpr (std::is_same_v<ContactSetType, PointTriangleContactSet>)
    {
        return std::make_pair(mOgcState.mPointGeometryPrefix, mOgcState.mTriangleGeometryPrefix);
    }
    else if constexpr (std::is_same_v<ContactSetType, EdgeEdgeContactSet>)
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
    Eigen::MatrixBase<TDerivedx> const& x_,
    TIndex i,
    int g,
    auto&& xi) const
{
    using EGeometry = decltype(mOgcState)::EGeometry;
    using math::linalg::mini::FromEigen;
    static auto constexpr kDims = std::decay_t<decltype(xi)>::kRows;
    auto x                      = x_.reshaped();
    i -= mOgcState.mPointGeometryPrefix[g];
    switch (g)
    {
        case EGeometry::Dynamic: xi = FromEigen(x.template segment<kDims>(i * kDims)); break;
        case EGeometry::Static: xi = FromEigen(mXstatic.col(i).template topRows<kDims>()); break;
        default: break;
    }
    return i + mOgcState.mPointGeometryPrefix[g];
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedx>
inline auto MeshDynamics<TScalar, TIndex>::LoadHalfEdge(
    Eigen::MatrixBase<TDerivedx> const& x_,
    TIndex he,
    int g,
    auto&& xi,
    auto&& xj) const
{
    using EGeometry = decltype(mOgcState)::EGeometry;
    using math::linalg::mini::FromEigen;
    static auto constexpr kDims = std::decay_t<decltype(xi)>::kRows;
    auto x                      = x_.reshaped();
    he -= mOgcState.mHalfEdgeGeometryPrefix[g];
    TIndex i, j;
    switch (g)
    {
        case EGeometry::Dynamic: {
            i  = geometry::IncomingVertex(*mOgcInput.F, he);
            j  = geometry::OutgoingVertex(*mOgcInput.F, he);
            xi = FromEigen(x.template segment<kDims>(i * kDims));
            xj = FromEigen(x.template segment<kDims>(j * kDims));
        }
        break;
        case EGeometry::Static: {
            i  = geometry::IncomingVertex(*mOgcInput.Fenv, he);
            j  = geometry::OutgoingVertex(*mOgcInput.Fenv, he);
            xi = FromEigen(mXstatic.col(i).template topRows<kDims>());
            xj = FromEigen(mXstatic.col(j).template topRows<kDims>());
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
    Eigen::MatrixBase<TDerivedx> const& x_,
    TIndex f,
    int g,
    auto&& xi,
    auto&& xj,
    auto&& xk) const
{
    using EGeometry = decltype(mOgcState)::EGeometry;
    using math::linalg::mini::FromEigen;
    static auto constexpr kDims = std::decay_t<decltype(xi)>::kRows;
    auto x                      = x_.reshaped();
    f -= mOgcState.mTriangleGeometryPrefix[g];
    TIndex i, j, k;
    switch (g)
    {
        case EGeometry::Dynamic: {
            auto xinds = mOgcInput.F->col(f);
            i          = xinds(0);
            j          = xinds(1);
            k          = xinds(2);
            xi         = FromEigen(x.template segment<kDims>(i * kDims));
            xj         = FromEigen(x.template segment<kDims>(j * kDims));
            xk         = FromEigen(x.template segment<kDims>(k * kDims));
        }
        break;
        case EGeometry::Static: {
            auto xinds = mOgcInput.Fenv->col(f);
            i          = xinds(0);
            j          = xinds(1);
            k          = xinds(2);
            xi         = FromEigen(mXstatic.col(i).template topRows<kDims>());
            xj         = FromEigen(mXstatic.col(j).template topRows<kDims>());
            xk         = FromEigen(mXstatic.col(k).template topRows<kDims>());
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
inline auto MeshDynamics<TScalar, TIndex>::LoadPoint(TIndex i, [[maybe_unused]] int g) const
{
    // This is trivial, but we just keep it for API consistency
    return i;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::LoadHalfEdge(TIndex he, int g) const
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
inline auto MeshDynamics<TScalar, TIndex>::LoadTriangle(TIndex f, int g) const
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
template <int kDims, int kStencil, class TDerivedH>
inline TScalar MeshDynamics<TScalar, TIndex>::RayleighQuotient(
    Eigen::SparseCompressedBase<TDerivedH> const& H,
    math::linalg::mini::SVector<TScalar, kStencil * kDims> const& gradc,
    std::array<TIndex, kStencil> const& nodes) const
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
            using InnerIteratorType = typename std::remove_cvref_t<decltype(H)>::InnerIterator;
            InnerIteratorType it(H, i);
            for (TIndex knj = 0; knj < nNodes; ++knj)
            {
                TIndex j = nodes[knj] * kDims;
                while (it and it.index() < j)
                    ++it;
                auto jend = j + kDims;
                for (; it and it.index() < jend; ++it)
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
