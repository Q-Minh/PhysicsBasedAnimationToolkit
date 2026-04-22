/**
 * @file Core.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Core VBD API.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_CORE_H
#define PBAT_SIM_ALGORITHM_VBD_CORE_H

#include "Enums.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Enums.h"
#include "pbat/io/Archive.h"
#include "pbat/math/LogInterpolate.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/algorithm/common/Common.h"
#include "pbat/sim/algorithm/vbd/Kernels.h"
#include "pbat/sim/contact/MeshDynamics.h"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/core.h>
#include <limits>
#include <tbb/parallel_for.h>
#include <tuple>

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Construct vertex-element adjacency graph
 *
 * @param E Element connectivity `|# nodes per element| x |# elements|` array
 * @param nNodes Number of nodes in the mesh
 * @param GVGp `|# verts+1|` prefixes into GVGe
 * @param GVGe `|# of vertex-elems adjacencies|` element indices s.t. `GVGe[k] for GVGp[i] <= k <
 * GVGp[i+1]` gives the element `e` adjacent to vertex `i`
 * @param GVGilocal `|# of vertex-elems adjacencies|` local vertex indices s.t. `GVGilocal[k] for
 * GVGp[i] <= k < GVGp[i+1]` gives the element `e` adjacent to vertex `i` GVGp[i+1]` gives the local
 * vertex index of vertex `i` in element `e=GVGe[k]`
 */
PBAT_API void VertexElementAdjacencyGraph(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    IndexVectorX& GVGp,
    IndexVectorX& GVGe,
    IndexVectorX& GVGilocal);

/**
 * @brief Compute vertex colors using a greedy algorithm
 *
 * @param E `|# nodes per element| x |# elements|` element connectivity array
 * @param nNodes Number of nodes in the mesh
 * @param eOrdering Vertex color ordering strategy
 * @param eSelection Vertex color selection strategy
 * @param GVVp `|# verts+1|` prefixes into GVVadj
 * @param GVVadj `|# vertex-vertex adjacencies|` adjacent vertex indices
 * @param colors `|# verts| x 1` Vertex colors
 */
PBAT_API void VertexColors(
    Eigen::Ref<IndexMatrixX const> const& E,
    Index nNodes,
    graph::EGreedyColorOrderingStrategy eOrdering,
    graph::EGreedyColorSelectionStrategy eSelection,
    IndexVectorX& GVVp,
    IndexVectorX& GVVadj,
    IndexVectorX& colors);

/**
 * @brief VBD simulation configuration
 * @details See \cite anka2024vbd
 */
struct Params
{
  public:
    /**
     * @brief Vertex-element adjacency graph
     * @param _GVGp `|# verts+1|` prefixes into GVGe
     * @param _GVGe `|# of vertex-elems adjacencies|` element indices s.t. `GVGe[k] for GVGp[i] <= k
     * < GVGp[i+1]` gives the element `e` adjacent to vertex `i`
     * @param _GVGilocal `|# of vertex-elems adjacencies|` local vertex indices s.t. `GVGilocal[k]
     * for GVGp[i] <= k < GVGp[i+1]` gives the local vertex index of vertex `i` in element
     * `e=GVGe[k]`
     * @return Reference to this
     */
    PBAT_API Params& WithVertexElementAdjacencyGraph(
        Eigen::Ref<IndexVectorX const> const& _GVGp,
        Eigen::Ref<IndexVectorX const> const& _GVGe,
        Eigen::Ref<IndexVectorX const> const& _GVGilocal);
    /**
     * @brief Vertex colors
     * @param _GVVp `|# verts+1|` prefixes into GVVadj
     * @param _GVVadj `|# vertex-vertex adjacencies|` adjacent vertex indices
     * @param _colors Vertex colors
     * @return Reference to this
     */
    PBAT_API Params& WithVertexColors(
        Eigen::Ref<IndexVectorX const> const& _GVVp,
        Eigen::Ref<IndexVectorX const> const& _GVVadj,
        Eigen::Ref<IndexVectorX const> const& _colors);
    /**
     * @brief Rayleigh damping coefficient
     * @param _betaR Rayleigh damping coefficient
     * @return Reference to this
     */
    PBAT_API Params& WithDamping(Scalar _betaR);
    /**
     * @brief Maximum number of outer iterations
     * @param nIters Maximum number of outer iterations
     * @return Reference to this
     */
    PBAT_API Params& WithMaximumIterations(Index nIters);
    /**
     * @brief Maximum number of VBD iterations per subproblem
     * @param nIters Maximum number of VBD iterations
     * @return Reference to this
     */
    PBAT_API Params& WithSubproblemMaximumIterations(Index nIters);
    /**
     * @brief Stencil gradient acceleration parameters
     * @param betaG0 Initial augmentation coefficient `0 < betaG0 < 1`
     * @param rhohat Lipschitz-normalized threshold above which steps are considered small (i.e.
     * solver progress is slow)
     * @param gammadown Beta reduction factor
     * @param gammaup Beta increase factor
     * @param wkinetic Weight factor for kinetic energy
     * @param welastic Weight factor for elastic energy
     * @param wcontact Weight factor for contact energy
     * @return Reference to this
     */
    PBAT_API Params& WithStencilGradientAcceleration(
        Scalar betaG0,
        Scalar rhohat,
        Scalar gammadown,
        Scalar gammaup,
        Scalar wkinetic = Scalar(1),
        Scalar welastic = Scalar(1),
        Scalar wcontact = Scalar(1));
    /**
     * @brief Vertex linear solver
     * @param solver Vertex integration linear solver
     * @param zero Numerical zero for hessian singularity check
     * @param eps Vertex integration linear solver epsilon
     * @param iters Maximum number of vertex integration linear solver iterations
     * @return Reference to this
     */
    PBAT_API Params& WithVertexLinearSolver(
        EVertexIntegrationLinearSolver solver,
        Scalar zero = std::numeric_limits<Scalar>::epsilon(),
        Scalar eps  = std::numeric_limits<Scalar>::epsilon(),
        int iters   = -1);
    /**
     * @brief Construct the simulation data
     * @param bValidate Throw on detected ill-formed inputs
     * @return Reference to this
     */
    PBAT_API Params& Construct(bool bValidate = true);
    /**
     * @brief Serialize this to archive
     * @param archive Archive to serialize to
     * @param bMinimal If true, only serialize stateless configuration parameters (scalars, enums).
     * If false, also serialize solver state and mesh-dependent data.
     */
    PBAT_API void Serialize(io::Archive& archive, bool bMinimal = true) const;
    /**
     * @brief Deserialize this from archive
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive const& archive);

  public:
    // Vertex-element adjacency graph
    IndexVectorX GVGp;      ///< `|# verts+1|` prefixes into GVGe
    IndexVectorX GVGe;      ///< `|# of vertex-elems adjacencies|` element indices s.t.
                            ///< `GVGe[k] for GVGp[i] <= k < GVGp[i+1]` gives the element `e`
                            ///< adjacent to vertex `i`
    IndexVectorX GVGilocal; ///< `|# of vertex-elems adjacencies|` local vertex indices s.t.
                            ///< `GVGilocal[k] for GVGp[i] <= k < GVGp[i+1]` gives the local vertex
                            ///< index of vertex `i` in element `e=GVGe[k]`
    // Parallelization
    IndexVectorX colors; ///< `|# vertices|` map of vertex colors
    IndexVectorX GVVp;   ///< `|# verts+1|` prefixes into GVVadj
    IndexVectorX GVVadj; ///< `|# vertex-vertex adjacencies|` adjacent vertex indices
    IndexVectorX Pptr;   ///< `|# partitions+1|` partition pointers, s.t. the range `[Pptr[p],
                         ///< Pptr[p+1])` indexes into Padj from partition `p`
    IndexVectorX Padj;   ///< `|# verts|` partition vertices
    Scalar betaR{0};     ///< Rayleigh damping coefficient

    // Convergence
    Index nMaxIters{20};           ///< Maximum number of outer augmented Lagrangian iterations
    Index nSubproblemMaxIters{25}; ///< Maximum number of VBD iterations per subproblem
    Scalar gtol{1e-3};             ///< Gradient norm convergence threshold

    // Vertex solve
    Scalar hessZero{
        std::numeric_limits<Scalar>::epsilon()}; ///< Numerical zero for hessian singularity check
    EVertexIntegrationLinearSolver eSolver{
        EVertexIntegrationLinearSolver::Inverse}; ///< Vertex integration linear solver
    Scalar vLinSolverEps{
        std::numeric_limits<Scalar>::epsilon()}; ///< Vertex integration linear solver epsilon
    int vLinSolverMaxIters{25}; ///< Maximum number of vertex integration linear solver iterations

    // Stencil gradient acceleration
    Scalar betaG0{0.5};   ///< Initial stencil gradient augmentation coefficient `0 < betaG0 < 1`
    Scalar rhohat{0.005}; ///< Lipschitz-normalized threshold above which steps are considered small
                          ///< (i.e. solver progress is slow)
    Scalar gammadown{0.95}; ///< Beta reduction factor
    Scalar gammaup{0.5};    ///< Beta increase factor
    Scalar wkinetic{1};     ///< Stencil gradient weight for kinetic energy term
    Scalar welastic{1};     ///< Stencil gradient weight for elastic energy term
    Scalar wcontact{1};     ///< Stencil gradient weight for contact energy term

    /**
     * @brief Read-write
     */
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> xb; ///< `3 x |# nodes|` buffer positions
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
        gkinetic; ///< `3 x |# nodes|` kinetic energy gradient at iteration k
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
        gelastic; ///< `3 x |# nodes|` elastic energy gradient at iteration k
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
        gcontact; ///< `3 x |# nodes|` contact energy gradient at iteration k
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> xk; ///< `3 x |# nodes|` past iteration
    Eigen::Vector<Scalar, Eigen::Dynamic> Hnk;   ///< Hessian norms at iteration k
    Eigen::Vector<Scalar, Eigen::Dynamic>
        betaG; ///< Per-vertex stencil gradient augmentation scale coefficient `0 < betaG < 1`
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> Hk; ///< `3 x 3*|# nodes|` block-diagonal Hessian,
                                                 ///< stored as contiguous 3x3 blocks per vertex
    Index k;                                     ///< Current iteration index
    Index kp;                                    ///< Current subproblem iteration index
};

/**
 * @brief Initialize VBD solve by preparing contact displacement bounds
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics (in/out parameter)
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params);

/**
 * @brief One VBD minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
/**
 * @brief Assemble a block-diagonal approximation of the dynamics Hessian (elastic + momentum +
 * damping, without contacts) into `params.Hk`.
 *
 * `params.Hk` is a `3 x 3*nNodes` matrix where the 3x3 block for vertex `i` is stored in
 * columns `[3*i, 3*i+3)`.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void AssembleBlockDiagonalDynamicsHessian(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    Params& params);

/**
 * @brief Compute the penalty parameter for the current subproblem
 * @param contact Mesh contact dynamics
 * @param params Solver parameters
 */
void UpdatePenaltyParameter(contact::MeshDynamics<Scalar, Index>& contact, Params const& params);

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params);

/**
 * @brief Integrate FEM elasto dynamics one step using VBD as the non-linear solver
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics (in/out parameter)
 * @param params Solver parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params);

namespace detail {

/// @brief Check if any component of a 3-vector is NaN or Inf
inline bool HasNonFinite(math::linalg::mini::SVector<Scalar, 3> const& v)
{
    return not std::isfinite(v(0)) or not std::isfinite(v(1)) or not std::isfinite(v(2));
}

/// @brief Check if any component of a 3x3 matrix is NaN or Inf
inline bool HasNonFinite(math::linalg::mini::SMatrix<Scalar, 3, 3> const& M)
{
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            if (not std::isfinite(M(r, c)))
                return true;
    return false;
}

/**
 * @brief Accumulate elastic energy derivatives for vertex i
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @tparam FOnEnergyDerivativesComputed Callable with signature `void(Scalar lambda, Scalar mu,
 * mini::SVector<Scalar,3> const& gi, mini::SMatrix<Scalar,3,3> const& Hi)`
 * @param i Vertex index
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 * @param gi `3 x 1` gradient accumulator for vertex `i`
 * @param Hi `3 x 3` Hessian accumulator for vertex `i`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void AccumulateElasticEnergy(
    Index i,
    common::FemElastoDynamics<TElasticEnergy>& fem,
    Params& params,
    math::linalg::mini::SVector<Scalar, 3>& gi,
    math::linalg::mini::SMatrix<Scalar, 3, 3>& Hi)
{
    using namespace math::linalg;
    using mini::FromEigen;
    auto begin = params.GVGp(i);
    auto end   = params.GVGp(i + 1);
    for (auto n = begin; n < end; ++n)
    {
        auto ilocal                     = params.GVGilocal(n);
        auto e                          = params.GVGe(n);
        auto lamee                      = fem.lamegU.col(e);
        auto wg                         = fem.wgU(e);
        auto ti                         = fem.mesh.E.col(e);
        mini::SMatrix<Scalar, 4, 3> GPe = FromEigen(fem.GNegU.template block<4, 3>(0, e * 3));
        mini::SMatrix<Scalar, 3, 4> xe =
            FromEigen(fem.x(Eigen::placeholders::all, ti).template block<3, 4>(0, 0));
        mini::SMatrix<Scalar, 3, 3> Fe = xe * GPe;
        TElasticEnergy Psi{};
        mini::SVector<Scalar, 9> gF;
        mini::SMatrix<Scalar, 9, 9> HF;
        Psi.GradAndHessian(Fe, lamee(0), lamee(1), gF, HF);
        kernels::AccumulateElasticHessian(ilocal, wg, GPe, HF, Hi);
        kernels::AccumulateElasticGradient(ilocal, wg, GPe, gF, gi);
    }
}

/**
 * @brief Accumulate augmented Lagrangian contact energy derivatives for vertex i.
 *
 * Iterates over all contact stencils involving vertex `i`:
 *   - Forward point contacts (PointPoint, PointEdge, PointTriangle) where `i` is the source
 *   - Reverse edge/triangle contacts via incident half-edges of `i` on the dynamic mesh
 *   - Edge-edge contacts on half-edges incident on `i`
 *
 * For each stencil node position `ki` where vertex `i` appears, accumulates:
 *   - Normal gradient:  `(kn*cs - lambda) * gradc_i`
 *   - Normal Hessian:   `kn * gradc_i * gradc_i^T`
 *   - Friction gradient: `W(ki) * T * (kf*cf - lambdaf)`
 *   - Friction Hessian:  `kf * W(ki)^2 * T * T^T`
 *
 * @tparam TDerivedx Type of position matrix
 * @param i Vertex index
 * @param x `3 x |# nodes|` current position matrix
 * @param contact Mesh contact dynamics (with linearized constraints)
 * @param gi Per-vertex gradient accumulator (3 x 1)
 * @param Hi Per-vertex Hessian accumulator (3 x 3)
 */
template <class TDerivedx>
inline void AccumulateContactEnergy(
    Index i,
    Eigen::MatrixBase<TDerivedx> const& x,
    contact::MeshDynamics<Scalar, Index> const& contact,
    math::linalg::mini::SVector<Scalar, 3>& gi,
    math::linalg::mini::SMatrix<Scalar, 3, 3>& Hi)
{
    auto const& contactParams        = contact.GetParams();
    Scalar const kn                  = contactParams.gamma * contactParams.kc;
    Scalar const kf                  = contactParams.gammaf * contactParams.kc;
    Scalar const dmin                = contactParams.dmin;
    auto fAccumulateNodalDerivatives = [&](auto C, auto stencil) {
        using ConstraintAccessorType   = decltype(C);
        using ContactSetType           = typename ConstraintAccessorType::ContactSetType;
        static auto constexpr kDofs    = ConstraintAccessorType::kDofs;
        static auto constexpr kStencil = ConstraintAccessorType::kStencil;
        auto const [Xc, nodes]         = contact.template LoadStencil<ContactSetType>(x, stencil);
        using namespace math::linalg;
        auto xc = mini::Reshape<kDofs, 1>(Xc);
        // Normal
        Scalar cs = C.Eval(xc) - dmin - C.Slack();
        Scalar dL = kn * cs - C.Lambda();
        // Friction
        auto F                      = C.Friction();
        auto const& Wf              = F.Weights();
        auto const& Tf              = F.TangentBasis();
        auto cf                     = F.Eval(xc);
        mini::SVector<Scalar, 2> df = kf * cf - F.Lambda();
        // Fetch local node index in the stencil
        Index ki{0};
        pbat::common::ForRange<0, kStencil>([&]<auto kj>() { ki += (i == nodes[kj]) * kj; });
        // Compute node derivatives
        auto const& gradc = C.Grad();
        kernels::AccumulateAugmentedLagrangianContactNodeDerivatives<
            3>(gradc, ki, dL, kn, Tf, Wf(ki), kf, df, C.Decay(), gi, Hi);
    };
    contact.ForEachPointPointContact(i, [&](auto C, auto stencil) {
        fAccumulateNodalDerivatives(C, stencil);
    });
    contact.ForEachPointEdgeContact(i, [&](auto C, auto stencil) {
        fAccumulateNodalDerivatives(C, stencil);
    });
    contact.ForEachPointTriangleContact(i, [&](auto C, auto stencil) {
        fAccumulateNodalDerivatives(C, stencil);
    });
    auto const& dm     = contact.DynamicMeshes();
    auto const hebegin = dm.GVHEp(i);
    auto const heend   = dm.GVHEp(i + 1);
    for (auto k = hebegin; k < heend; ++k)
    {
        auto const he = dm.GVHEadj(k);
        auto const f  = geometry::FaceOfHalfEdge(he);
        contact.ForEachEdgePointContact(he, [&](auto C, auto stencil) {
            fAccumulateNodalDerivatives(C, stencil);
        });
        contact.ForEachTrianglePointContact(f, [&](auto C, auto stencil) {
            fAccumulateNodalDerivatives(C, stencil);
        });
        contact.ForEachEdgeEdgeContact(he, [&](auto C, auto stencil) {
            fAccumulateNodalDerivatives(C, stencil);
        });
    }
}

/**
 * @brief Build the local vertex equation (gradient and Hessian) for vertex i, decomposed by energy
 * term.
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param i Vertex index
 * @param xi Position of vertex i
 * @param xti Position of vertex i at time t
 * @param xtildei Inertial target position of vertex i
 * @param m Mass of vertex i
 * @param h Time step size
 * @param h2 Time step size squared
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact dynamics
 * @param params Solver parameters
 * @return (Hi, gkinetici, gelastici, gcontacti) where Hi is the Hessian and the g vectors are
 * per-energy-term gradients for vertex i. The kinetic gradient includes the Rayleigh damping
 * contribution.
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
auto BuildVertexEquation(
    Index i,
    math::linalg::mini::SVector<Scalar, 3> const& xi,
    math::linalg::mini::SVector<Scalar, 3> const& xti,
    math::linalg::mini::SVector<Scalar, 3> const& xtildei,
    Scalar m,
    Scalar h,
    Scalar h2,
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
    -> std::tuple<
        math::linalg::mini::SMatrix<Scalar, 3, 3>,
        math::linalg::mini::SVector<Scalar, 3>,
        math::linalg::mini::SVector<Scalar, 3>,
        math::linalg::mini::SVector<Scalar, 3>>
{
    using namespace math::linalg;
    mini::SMatrix<Scalar, 3, 3> Hi = mini::Zeros<Scalar, 3, 3>();
    // Elastic energy gradient
    mini::SVector<Scalar, 3> gelastici = mini::Zeros<Scalar, 3, 1>();
    AccumulateElasticEnergy<TElasticEnergy>(i, fem, params, gelastici, Hi);
    gelastici *= h2;
    Hi *= h2;
    // Contact energy gradient (augmented Lagrangian)
    mini::SVector<Scalar, 3> gcontacti = mini::Zeros<Scalar, 3, 1>();
    AccumulateContactEnergy(i, params.xb, contact, gcontacti, Hi);
    // Kinetic energy gradient (+ Rayleigh damping, which couples through the full Hessian)
    mini::SVector<Scalar, 3> gkinetici = mini::Zeros<Scalar, 3, 1>();
    kernels::AddInertiaDerivatives(Scalar(1) /*h2*/, m, xtildei, xi, gkinetici, Hi);
    return {Hi, gkinetici, gelastici, gcontacti};
}

/**
 * @brief Adapt stencil gradient acceleration parameter for vertex i and compute the stencil
 * gradient augmentation.
 * @param i Vertex index
 * @param xi Current position of vertex i
 * @param gi Total gradient at vertex i
 * @param Hi Hessian at vertex i
 * @param gkinetici Kinetic energy gradient at vertex i (including Rayleigh damping)
 * @param gelastici Elastic energy gradient at vertex i
 * @param gcontacti Contact energy gradient at vertex i
 * @param params Solver parameters (in/out: betaG, gkinetic, gelastic, gcontact, xk, Hnk are
 * updated)
 * @return Augmentation vector to be added to the gradient
 */
inline math::linalg::mini::SVector<Scalar, 3> ComputeStencilGradientAugmentation(
    Index i,
    math::linalg::mini::SVector<Scalar, 3> const& xi,
    math::linalg::mini::SMatrix<Scalar, 3, 3> const& Hi,
    math::linalg::mini::SVector<Scalar, 3> const& gkinetici,
    math::linalg::mini::SVector<Scalar, 3> const& gelastici,
    math::linalg::mini::SVector<Scalar, 3> const& gcontacti,
    Params& params)
{
    using namespace math::linalg;
    using mini::Dot;
    using mini::FromEigen;
    using mini::Norm;
    using mini::ToEigen;
    // Adapt stencil gradient acceleration parameter using the total gradient
    Scalar constexpr kSmallEpsilon{1e-10};
    mini::SVector<Scalar, 3> gi = gkinetici + gelastici + gcontacti;
    if (params.kp > 0)
    {
        Scalar ngk                    = Norm(gi);
        mini::SVector<Scalar, 3> gkm1 = FromEigen(params.gkinetic.col(i).template head<3>()) +
                                        FromEigen(params.gelastic.col(i).template head<3>()) +
                                        FromEigen(params.gcontact.col(i).template head<3>());
        Scalar ngkm1                  = Norm(gkm1);
        Scalar ndgkm1                 = Norm(gi - gkm1);
        auto xk                       = params.xk.col(i).template head<3>();
        Scalar ndxkm1                 = std::max(Norm(xi - FromEigen(xk)), kSmallEpsilon);
        Scalar L                      = params.Hnk(i) + ngk / ndxkm1;
        Scalar rho                    = ndgkm1 / std::max(L * ndxkm1, kSmallEpsilon);
        if (ngk > ngkm1)
            params.betaG(i) *= params.gammadown;
        else if (rho > params.rhohat)
            params.betaG(i) += (1 - params.betaG(i)) * params.gammaup;
    }
    params.gkinetic.col(i) = ToEigen(gkinetici);
    params.gelastic.col(i) = ToEigen(gelastici);
    params.gcontact.col(i) = ToEigen(gcontacti);
    params.xk.col(i)       = ToEigen(xi);
    params.Hnk(i)          = Norm(Hi);
    // Compute weighted stencil gradient augmentation
    auto nbegin                 = params.GVVp(i);
    auto nend                   = params.GVVp(i + 1);
    mini::SVector<Scalar, 3> gp = mini::Zeros<Scalar, 3, 1>();
    for (auto n = nbegin; n < nend; ++n)
    {
        auto j = params.GVVadj(n);
        gp +=
            params.wkinetic * gkinetici + params.welastic * gelastici + params.wcontact * gcontacti;
    }
    Scalar lambda = params.betaG(i) * Dot(gi, gp) / std::max(Dot(gp, gp), kSmallEpsilon);
    return gi + std::max(lambda, Scalar(0)) * gp;
}

} // namespace detail

template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Iterate");
    auto h                 = fem.bdf.BetaTilde();
    auto h2                = h * h;
    auto xtildeBdf         = fem.bdf.Inertia(0).reshaped(fem.x.rows(), fem.x.cols());
    auto xt                = -xtildeBdf;
    params.xb              = fem.x; // Copy current positions to buffer
    auto const nPartitions = params.Pptr.size() - 1;
    for (Index p = 0; p < nPartitions; ++p)
    {
        auto const pBegin = params.Pptr(p);
        auto const pEnd   = params.Pptr(p + 1);
        tbb::parallel_for(pBegin, pEnd, [&](Index k) {
            // Solve vertex i
            using namespace math::linalg;
            using mini::FromEigen;
            using mini::ToEigen;
            auto i = params.Padj(k);
            if (fem.IsDirichletNode(i))
                return;
            // Fetch vertex data
            mini::SVector<Scalar, 3> xi      = FromEigen(fem.x.col(i).template head<3>());
            mini::SVector<Scalar, 3> xti     = FromEigen(xt.col(i).template head<3>());
            Scalar m                         = fem.m(i);
            mini::SVector<Scalar, 3> xtildei = FromEigen(fem.xtilde.col(i).template head<3>());
            // Compute vertex derivatives
            auto [Hi, gkinetici, gelastici, gcontacti] = detail::BuildVertexEquation<
                TElasticEnergy>(i, xi, xti, xtildei, m, h, h2, fem, contact, params);
            // kernels::AddDamping(Scalar(1) / h, xti, xi, params.betaR, gi, Hi);
            // Augment gradient with stencil gradient acceleration
            mini::SVector<Scalar, 3> gi = detail::ComputeStencilGradientAugmentation(
                i,
                xi,
                Hi,
                gkinetici,
                gelastici,
                gcontacti,
                params);
            // Solve
            kernels::IntegratePositions(
                gi,
                Hi,
                xi,
                params.eSolver,
                params.hessZero,
                params.vLinSolverEps,
                params.vLinSolverMaxIters);
            if (detail::HasNonFinite(xi))
            {
                fmt::print(stderr, "NaN/Inf after IntegratePositions for vertex {}\n", i);
                fmt::print(stderr, "gi: {}, {}, {}\n", gi(0), gi(1), gi(2));
                fmt::print(stderr, "Hi:\n");
                for (int r = 0; r < 3; ++r)
                    fmt::print(stderr, "{}, {}, {}\n", Hi(r, 0), Hi(r, 1), Hi(r, 2));
                throw std::runtime_error("NaN/Inf after IntegratePositions");
            }
            fem.x.col(i) = ToEigen(xi);
        });
    }
    ++params.kp;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.InitializeSolve");
    auto const xt = fem.bdf.CurrentState().reshaped(fem.x.rows(), fem.x.cols());
    contact.GetParams().ComputeQueryRadius((fem.xtilde - xt).colwise().norm().maxCoeff());
    contact.UpdateConstraintSet(xt, true /*bComputeReverseContactPairs*/);
    contact.RestoreFeasibility(fem.x, fem.dmask);
    params.k  = 0;
    params.xk = fem.x;
    params.betaG.setConstant(params.betaG0);
}

/**
 * @brief Check convergence of VBD solve
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact dynamics
 * @param params Solver parameters
 * @return `true` if converged, `false` otherwise
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
bool CheckConvergence(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index> const& contact,
    Params& params)
{
    auto bt  = fem.bdf.BetaTilde();
    auto bt2 = bt * bt;
    // Use gkinetic as scratch buffer for the assembled total gradient
    auto& gk = params.gkinetic;
    gk.setZero();
    fem.ToElasticGradient(fem.x, gk);
    gk *= bt2;
    fem.ToMomentumGradient(fem.x, gk);
    contact.ToGradient(fem.x, gk);
    auto gknorm2 = gk.squaredNorm();
    return gknorm2 <= params.gtol * params.gtol;
}

/**
 * @brief Compute 3x3 dynamics hessian blocks
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void AssembleBlockDiagonalDynamicsHessian(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.AssembleBlockDiagonalDynamicsHessian");
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::ToEigen;
    auto const nNodes = fem.x.cols();
    params.Hk.setZero();
    auto h         = fem.bdf.BetaTilde();
    auto h2        = h * h;
    auto xtildeBdf = fem.bdf.Inertia(0).reshaped(fem.x.rows(), fem.x.cols());
    auto xt        = -xtildeBdf;
    tbb::parallel_for(Index{0}, nNodes, [&](Index i) {
        if (fem.IsDirichletNode(i))
            return;
        mini::SVector<Scalar, 3> xi      = FromEigen(fem.x.col(i).template head<3>());
        mini::SVector<Scalar, 3> xti     = FromEigen(xt.col(i).template head<3>());
        Scalar m                         = fem.m(i);
        mini::SVector<Scalar, 3> xtildei = FromEigen(fem.xtilde.col(i).template head<3>());
        mini::SMatrix<Scalar, 3, 3> Hi   = mini::Zeros<Scalar, 3, 3>();
        mini::SVector<Scalar, 3> gi      = mini::Zeros<Scalar, 3, 1>();
        // Elastic energy
        detail::AccumulateElasticEnergy<TElasticEnergy>(i, fem, params, gi, Hi);
        Hi *= h2;
        // Kinetic energy
        kernels::AddInertiaDerivatives(Scalar(1) /*h2*/, m, xtildei, xi, gi, Hi);
        // Damping
        kernels::AddDamping(Scalar(1) / h, xti, xi, params.betaR, gi, Hi);
        params.Hk.template block<3, 3>(0, 3 * i) = ToEigen(Hi);
    });
}

template <physics::CHyperElasticEnergy TElasticEnergy, class TDerivedXt>
void GloballyRestoreFeasibility(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params const& params,
    Eigen::MatrixBase<TDerivedXt> const& xt)
{
    auto const dmax = (fem.x - xt).colwise().norm().maxCoeff();
    auto const dmin = contact.OgcState().bv.minCoeff();
    if (dmax > dmin)
    {
        fem.x(Eigen::placeholders::all, fem.FreeNodes()) =
            xt(Eigen::placeholders::all, fem.FreeNodes()) +
            (dmin / dmax) * (fem.x - xt)(Eigen::placeholders::all, fem.FreeNodes());
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Solve");
    for (params.k = 0; params.k < params.nMaxIters; ++params.k)
    {
        // 1. Linearize constraints
        auto xt = fem.bdf.CurrentState().reshaped(fem.x.rows(), fem.x.cols());
        contact.LinearizeConstraints(fem.x, xt);
        // 2. Convergence check
        bool const bConverged = CheckConvergence(fem, contact, params);
        if (bConverged)
            break;
        // 3. Setup subproblem
        AssembleBlockDiagonalDynamicsHessian(fem, params);
        UpdatePenaltyParameter(contact, params);
        // params.betaG.setConstant(params.betaG0);
        // 4. VBD solve the linear constraint subproblem
        using EDualVariable = typename contact::MeshDynamics<Scalar, Index>::EDualVariable;
        for (params.kp = 0; params.kp < params.nSubproblemMaxIters;)
        {
            contact.UpdateDual<EDualVariable::Slack>(fem.x);
            Iterate(fem, contact, params);
        }
        contact.UpdateDual<
            EDualVariable::Slack | EDualVariable::LagrangeMultiplier | EDualVariable::Decay>(fem.x);
        // 5. Restore feasibility
        contact.RestoreFeasibility(fem.x, fem.dmask);
        // GloballyRestoreFeasibility(fem, contact, params, contact.DynamicPointPositions());
        // 6. Update constraint set for next subproblem
        contact.UpdateConstraintSet(fem.x, true /*bComputeReverseContactPairs*/);
    }
    fem.BackSubstituteIntegratedPositionsIntoVelocities();
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    common::FemElastoDynamics<TElasticEnergy>& fem,
    contact::MeshDynamics<Scalar, Index>& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Integrate");
    Solve(fem, contact, params);
    fem.Step();
}

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_CORE_H
