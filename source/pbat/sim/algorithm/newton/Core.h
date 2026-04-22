#ifndef PBAT_SIM_ALGORITHM_NEWTON_CORE_H
#define PBAT_SIM_ALGORITHM_NEWTON_CORE_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/io/Archive.h"
#include "pbat/math/linalg/mini/Reductions.h"
#include "pbat/math/linalg/mini/Reshape.h"
#include "pbat/math/optimization/Newton.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/contact/MeshDynamics.h"
#include "pbat/sim/dynamics/FemElastoDynamics.h"

#ifdef PBAT_USE_SUITESPARSE
    #include "pbat/math/linalg/CholmodSupport.h"
#else
    #include <Eigen/SparseCholesky>
#endif // PBAT_USE_SUITESPARSE

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <exception>
#include <variant>

namespace pbat::sim::algorithm::newton {

/**
 * @brief Linear solver types for the Newton method
 */
enum class ELinearSolver {
    LLT,          ///< Cholesky LLT decomposition
    PCGJacobi,    ///< Preconditioned Conjugate Gradient with Jacobi (i.e. diagonal) preconditioner
    PCGIC,        ///< Preconditioned Conjugate Gradient with Incomplete Cholesky preconditioner
    PCGILUT,      ///< Preconditioned Conjugate Gradient with Incomplete LU with thresholding
                  ///< preconditioner
    PCGLaplacian, ///< Preconditioned Conjugate Gradient with Laplacian preconditioner
};

/**
 * @brief Truncation strategies for OGC-based contact
 */
enum class EOgcTruncationStrategy {
    PerVertex, ///< Truncate displacements per-vertex based on contact distance bounds
    Global,    ///< Truncate displacements globally based on contact distance bounds
};

/**
 * @brief Parameters for the Newton simulation algorithm.
 */
struct Params
{
    /**
     * @brief Set the Newton optimizer.
     * @param optimizer Newton optimizer to use
     * @return Reference to this
     */
    PBAT_API Params& WithOptimizer(math::optimization::Newton<Scalar> optimizer);
    /**
     * @brief Set the SPD correction method for hyper-elastic Hessians.
     * @param eSpdCorrectionIn SPD correction method
     * @return Reference to this
     */
    PBAT_API Params& WithSpdCorrection(fem::EHyperElasticSpdCorrection _eSpdCorrection);
    /**
     * @brief Set the linear solver type for the Newton step.
     * @param _eLinearSolver Linear solver type
     * @param maxIters Maximum number of iterations (for iterative solvers)
     * @param tol Tolerance (for iterative solvers)
     * @return Reference to this
     */
    PBAT_API Params&
    WithLinearSolver(ELinearSolver _eLinearSolver, Eigen::Index maxIters = 100, Scalar tol = 1e-6);
    /**
     * @brief Set the OGC truncation strategy.
     * @param strategy OGC truncation strategy
     * @return Reference to this
     */
    PBAT_API Params& WithOgcTruncationStrategy(EOgcTruncationStrategy strategy);
    /**
     * @brief Set the maximum number of iterations for the Newton solver.
     * @param n Maximum number of iterations
     * @return Reference to this
     */
    PBAT_API Params& WithMaxIters(std::int32_t n);
    /**
     * @brief Construct the parameters
     * @param bValidate Throw on detected ill-formed inputs
     * @return Reference to this
     */
    PBAT_API Params& Construct(bool bValidate = true);
    /**
     * @brief Serialize this to archive
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize this from archive
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive const& archive);

    std::int32_t nMaxIters{20};                ///< Maximum number of linear constraint subproblems
    math::optimization::Newton<Scalar> newton; ///< Newton optimizer
    std::vector<Eigen::Triplet<Scalar, Index>> triplets; ///< Triplets for assembling the Hessian
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index> hessian; ///< Hessian matrix
    fem::EHyperElasticSpdCorrection eSpdCorrection{
        fem::EHyperElasticSpdCorrection::Absolute};  ///< SPD correction method for hyper-elastic
                                                     ///< Hessians
    ELinearSolver eLinearSolver{ELinearSolver::LLT}; ///< Linear solver type for Newton step
    EOgcTruncationStrategy eOgcTruncationStrategy{
        EOgcTruncationStrategy::PerVertex}; ///< OGC truncation strategy

#ifdef PBAT_USE_SUITESPARSE
    using DecompositionType =
        Eigen::CholmodDecomposition<decltype(hessian), Eigen::Lower>; ///< Cholesky decomposition
                                                                      ///< type
#else
    using DecompositionType =
        Eigen::SimplicialLDLT<decltype(hessian)>; ///< Cholesky decomposition type
#endif // PBAT_USE_SUITESPARSE
    using IncompleteCholeskyType =
        Eigen::IncompleteCholesky<Scalar, Eigen::Lower, Eigen::AMDOrdering<Index>>;
    using IncompleteLUTType = Eigen::IncompleteLUT<Scalar, Index>;
    using SolverType = std::variant<
        DecompositionType,
        Eigen::ConjugateGradient<
            decltype(hessian),
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<Scalar>>,
        Eigen::ConjugateGradient<
            decltype(hessian),
            Eigen::Lower | Eigen::Upper,
            IncompleteCholeskyType>,
        Eigen::ConjugateGradient<
            decltype(hessian),
            Eigen::Lower | Eigen::Upper,
            IncompleteLUTType>/*,
        // TODO: Implement the Laplacian preconditioner
        Eigen::ConjugateGradient<
            decltype(hessian),
            Eigen::Lower | Eigen::Upper,
            fem::LaplacianPreconditioner<Scalar>>*/>;
    SolverType Hinv;   ///< Hessian inverse
    std::int32_t k{0}; ///< Current iteration number
};

/**
 * @brief Finite element elasto dynamics problem for Newton's method
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
using FemElastoDynamics =
    dynamics::FemElastoDynamics<fem::Tetrahedron<1>, 3, TElasticEnergy, Scalar, Index>;

/**
 * @brief Mesh dynamics problem for Newton's method
 */
using MeshDynamics = sim::contact::MeshDynamics<Scalar, Index>;

/**
 * @brief Truncate displacement vector based on OGC truncation strategy
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 * @param dxk Displacement vector to be truncated (in/out parameter)
 */
template <physics::CHyperElasticEnergy TElasticEnergy, class TDerivedDxk>
void TruncateDisplacement(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params,
    Eigen::MatrixBase<TDerivedDxk>& dxk);

/**
 * @brief Initialize the solve process for the given finite element elasto dynamics contact problem.
 *
 * Computes OGC query radius, updates the constraint set, restores feasibility, and resets
 * the outer iteration counter `params.k = 0`.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params);

/**
 * @brief Linearize constraints at the current iterate.
 *
 * Calls `contact.LinearizeConstraints(x)` to compute the linearized constraint data
 * (chat, gradc) for the current iterate.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @pre `InitializeSolve` has been called
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void LinearizeConstraints(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact);

/**
 * @brief Check KKT convergence of the outer (nonlinear) problem.
 *
 * Computes the full gradient (elastic + momentum + contact) into params.newton.gk and checks if the
 * squared gradient norm is below the convergence threshold `params.newton.gtol2`.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 * @return true if KKT conditions are satisfied (converged), false otherwise
 * @pre `LinearizeConstraints` has been called
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
bool CheckConvergence(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params);

/**
 * @brief Prepare a linearized constraint subproblem.
 *
 * Precomputes elastic energy derivatives, assembles the Hessian (without contact contributions),
 * updates barrier parameters, and initializes the inner Newton solver for the current subproblem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void PrepareSubproblem(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params);

/**
 * @brief Prepare next iteration of the current linearized constraint subproblem.
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 * @param bAreSubproblemDerivativesDirty Whether to compute subproblem derivatives
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void PrepareNextIteration(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params,
    bool bAreSubproblemDerivativesDirty = true);

/**
 * @brief One Newton iteration of the current linearized constraint subproblem.
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @return true if step was taken, false otherwise
 * @pre `PrepareSubproblem` or `PrepareNextIteration` has been called
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
bool Iterate(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params);

/**
 * @brief Finalize the current linearized constraint subproblem.
 *
 * Restores feasibility, updates the constraint set, and increments the outer iteration counter
 * `params.k`.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void FinalizeSubproblem(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using Newton's method
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @return true if converged, false otherwise
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
bool Solve(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params);

/**
 * @brief Compute the merit function value for the Newton optimization.
 *
 * Evaluates \f$ K + \beta^2 U + C \f$, where \f$ K \f$ is the momentum energy,
 * \f$ U \f$ is the hyper-elastic potential, and \f$ C \f$ is the contact potential.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @return Merit function value
 * @pre `ComputeElasticDerivatives()` has been called
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
Scalar MeritFunctionFromPrecomputedPotentials(
    FemElastoDynamics<TElasticEnergy> const& fem,
    MeshDynamics const& contact)
{
    Scalar bt  = fem.bdf.BetaTilde();
    Scalar bt2 = bt * bt;
    Scalar U   = fem::HyperElasticPotential(fem.UgU);
    Scalar K   = fem.MomentumEnergy(fem.x);
    Scalar C   = contact.Potential(fem.x);
    return K + bt2 * U + C;
}

/**
 * @brief Derivative precomputation for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void ComputeElasticDerivatives(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.ComputeElasticDerivatives");
    // Precompute elastic energy and its derivatives
    Scalar bt  = fem.bdf.BetaTilde();
    Scalar bt2 = bt * bt;
    fem.ComputeElasticEnergy(
        fem.x,
        fem::EElementElasticityComputationFlags::Potential |
            fem::EElementElasticityComputationFlags::Gradient |
            fem::EElementElasticityComputationFlags::Hessian,
        params.eSpdCorrection);
    fem.HgU *= bt2;
    fem.GgU *= bt2;
}

/**
 * @brief Derivative precomputation for the given finite element elasto dynamics problem.
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void ComputeFrictionDerivatives(FemElastoDynamics<TElasticEnergy> const& fem, MeshDynamics& contact)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.ComputeFrictionDerivatives");
    // Precompute friction energy and its derivatives
    auto xt = -fem.bdf.Inertia(0);
    auto bt = fem.bdf.BetaTilde();
    contact.ComputeEnergy(
        fem.x,
        xt,
        bt,
        MeshDynamics::EComputeFlags::Hessian,
        math::linalg::EEigenvalueFilter::SpdProjection);
}

/**
 * @brief Derivative precomputation for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param gk Gradient vector
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void ToGradient(
    FemElastoDynamics<TElasticEnergy> const& fem,
    MeshDynamics& contact,
    Eigen::Vector<Scalar, Eigen::Dynamic>& gk,
    bool bForSubproblem = false)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.ToGradient");
    if (bForSubproblem)
        contact.UpdateDual<MeshDynamics::EDualVariable::Slack>(fem.x);
    // Gradient of 1/2 |x - \Tilde{x}|_M^2 + bt^2 U(x) + bt^2 C(x)
    gk.setZero();
    fem::ToHyperElasticGradient(fem.mesh, fem.egU, fem.GgU, gk);
    contact.ToGradient(fem.x, gk);
    gk += ((fem.x - fem.xtilde) * fem.m.asDiagonal()).reshaped();
    gk(fem.DirichletDofs()).setZero();
}

/**
 * @brief Assemble the Hessian for the given finite element elasto dynamics problem.
 *
 * Assembles the Hessian consisting of mass + hyper-elastic contributions. When
 * `bWithContacts` is true, also adds the linearized contact barrier Hessian
 * contribution, which is a sum of rank-1 outer products
 * \f$ -\mu_i a''(\hat{c}_i + \nabla c_i^T x) \nabla c_i \nabla c_i^T \f$ per constraint.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 * @param bWithContacts Whether to include linearized contact barrier Hessian contribution
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void AssembleHessian(
    FemElastoDynamics<TElasticEnergy> const& fem,
    MeshDynamics const& contact,
    Params& params,
    bool bWithContacts = true)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.AssembleHessian");
    // Hessian of 1/2 |x - \Tilde{x}|_M^2 + bt^2 U(x) [+ bt^2 C(x)]
    auto constexpr kDims = std::remove_cvref_t<decltype(fem)>::kDims;
    auto nTriplets       = fem.HgU.size() + fem.M().size();
    if (bWithContacts)
    {
        auto constexpr kPointPointDofs =
            std::decay_t<decltype(contact.PointPointContacts())>::AccessorType::kDofs;
        auto constexpr kPointEdgeDofs =
            std::decay_t<decltype(contact.PointEdgeContacts())>::AccessorType::kDofs;
        auto constexpr kPointTriangleDofs =
            std::decay_t<decltype(contact.PointTriangleContacts())>::AccessorType::kDofs;
        auto constexpr kEdgeEdgeDofs =
            std::decay_t<decltype(contact.EdgeEdgeContacts())>::AccessorType::kDofs;
        auto const nPointPointContacts    = contact.PointPointContacts().Size();
        auto const nPointEdgeContacts     = contact.PointEdgeContacts().Size();
        auto const nPointTriangleContacts = contact.PointTriangleContacts().Size();
        auto const nEdgeEdgeContacts      = contact.EdgeEdgeContacts().Size();
        nTriplets += nPointPointContacts * kPointPointDofs * kPointPointDofs +
                     nPointEdgeContacts * kPointEdgeDofs * kPointEdgeDofs +
                     nPointTriangleContacts * kPointTriangleDofs * kPointTriangleDofs +
                     nEdgeEdgeContacts * kEdgeEdgeDofs * kEdgeEdgeDofs;
    }
    params.triplets.reserve(nTriplets);
    params.triplets.clear();
    // Mass matrix contribution
    for (Eigen::Index i = 0; i < fem.m.size(); ++i)
        for (auto d = 0; d < kDims; ++d)
            params.triplets.emplace_back(i * kDims + d, i * kDims + d, fem.m(i));
    // Hyper-elastic Hessian contribution
    using ElementType = typename FemElastoDynamics<TElasticEnergy>::ElementType;
    auto nQuadPtsU    = fem.egU.size();
    for (Eigen::Index g = 0; g < nQuadPtsU; ++g)
    {
        auto const nodes = fem.mesh.E.col(fem.egU(g));
        auto const HUg =
            fem.HgU.template block<kDims * ElementType::kNodes, kDims * ElementType::kNodes>(
                0,
                g * kDims * ElementType::kNodes);
        for (auto jl = 0; jl < ElementType::kNodes; ++jl)
            for (auto jd = 0; jd < kDims; ++jd)
                for (auto il = 0; il < ElementType::kNodes; ++il)
                    for (auto id = 0; id < kDims; ++id)
                        params.triplets.emplace_back(
                            nodes(il) * kDims + id,
                            nodes(jl) * kDims + jd,
                            HUg(il * kDims + id, jl * kDims + jd));
    }
    // Linearized contact barrier Hessian contribution:
    // \f$ \sum_c \gamma_c \mu \nabla c \nabla c^T \f$
    if (bWithContacts)
    {
        using math::linalg::mini::Dot;
        using math::linalg::mini::Reshape;
        using math::linalg::mini::ToEigen;
        auto const& contactParams        = contact.GetParams();
        Eigen::Index const nDynamicNodes = fem.x.size() / kDims;
        contact.ForAllContacts(
            [&]<class TContactSet>(
                typename TContactSet::ConstAccessorType C,
                typename MeshDynamics::Stencil stencil,
                std::int32_t /*t*/) {
                using ConstraintAccessorType   = decltype(C);
                auto nodes                     = contact.LoadStencil<TContactSet>(stencil);
                auto const& gradc              = C.Grad();
                auto gamma                     = /*C.Decay()*/ 1;
                auto mu                        = contactParams.kc;
                Scalar dH                      = gamma * mu;
                auto F                         = C.Friction();
                auto const& Hfu                = F.Hessian();
                auto const& Tf                 = F.TangentBasis();
                auto const& Wf                 = F.Weights();
                using SMatrixDD                = math::linalg::mini::SMatrix<Scalar, kDims, kDims>;
                SMatrixDD Hf                   = Tf * Hfu * Tf.Transpose();
                static auto constexpr kStencil = ConstraintAccessorType::kStencil;
                for (auto jl = 0; jl < kStencil; ++jl)
                {
                    if (nodes[jl] >= nDynamicNodes)
                        continue;
                    for (auto il = 0; il < kStencil; ++il)
                    {
                        if (nodes[il] >= nDynamicNodes)
                            continue;
                        Scalar wf = Wf(jl) * Wf(il);
                        for (auto jd = 0; jd < kDims; ++jd)
                            for (auto id = 0; id < kDims; ++id)
                                params.triplets.emplace_back(
                                    nodes[il] * kDims + id,
                                    nodes[jl] * kDims + jd,
                                    dH * gradc(il * kDims + id) * gradc(jl * kDims + jd) +
                                        wf * Hf(id, jd));
                    }
                }
            },
            1 /*nThreads*/);
    }
    // Assemble
    params.hessian.resize(fem.x.size(), fem.x.size());
    // Remove off-diagonal Dirichlet entries (always) and upper triangular part (when LLT is used)
    auto itRemoveBegin = std::remove_if(
        params.triplets.begin(),
        params.triplets.end(),
        [&](Eigen::Triplet<Scalar, Index> const& triplet) {
            bool bIsUpperTriangular = triplet.row() < triplet.col();
            bool bIsDiag            = triplet.row() == triplet.col();
            bool bIsDirichletEntry =
                fem.IsDirichletDof(triplet.row()) or fem.IsDirichletDof(triplet.col());
            return (bIsUpperTriangular and params.eLinearSolver == ELinearSolver::LLT) or
                   (not bIsDiag and bIsDirichletEntry);
        });
    params.triplets.erase(itRemoveBegin, params.triplets.end());
    params.hessian.setFromTriplets(params.triplets.begin(), params.triplets.end());
}

/**
 * @brief Compute the Hessian inverse product with the gradient for the given finite element elasto
 * dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param gk Gradient vector
 * @param dxk Search direction vector
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void HessianInverseProduct(
    Eigen::Vector<Scalar, Eigen::Dynamic> const& gk,
    Eigen::Vector<Scalar, Eigen::Dynamic>& dxk,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.HessianInverseProduct");
    // Compute inverse hessian product with gradient
    std::visit(
        [&](auto& solver) {
            solver.compute(params.hessian);
            switch (solver.info())
            {
                case Eigen::Success: break;
                case Eigen::ComputationInfo::NoConvergence:
                    throw std::runtime_error("No convergence in linear solver");
                case Eigen::ComputationInfo::NumericalIssue:
                    throw std::runtime_error("Numerical issue in linear solver");
                case Eigen::ComputationInfo::InvalidInput:
                    throw std::runtime_error("Invalid input to linear solver");
            }
            dxk = solver.solve(gk);
        },
        params.Hinv);
}

template <physics::CHyperElasticEnergy TElasticEnergy, class TDerivedDxk>
void TruncateDisplacement(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params,
    Eigen::MatrixBase<TDerivedDxk>& dxk)
{
    switch (params.eOgcTruncationStrategy)
    {
        case EOgcTruncationStrategy::PerVertex: {
            // Per-vertex truncation
            contact.MakeStepFeasible(dxk, fem.dmask);
            break;
        }
        case EOgcTruncationStrategy::Global: {
            // Global truncation
            auto const dmax = dxk.reshaped(fem.x.rows(), fem.x.cols()).colwise().norm().maxCoeff();
            auto const dmin = contact.OgcState().bv.minCoeff();
            if (dmax > dmin)
            {
                dxk *= (dmin / dmax);
                contact.RequestConstraintSetUpdate();
            }
            break;
        }
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy, class TDerivedXt>
void GloballyRestoreFeasibility(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
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

/**
 * @brief Truncate displaced positions based on OGC truncation strategy for solve initialization
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem The finite element elasto dynamics problem
 * @param contact The mesh dynamics
 * @param params The solver parameters
 * @param xt The current state
 */
template <physics::CHyperElasticEnergy TElasticEnergy, class TDerivedXt>
void RestoreFeasibility(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params const& params,
    Eigen::MatrixBase<TDerivedXt> const& xt)
{
    switch (params.eOgcTruncationStrategy)
    {
        case EOgcTruncationStrategy::PerVertex: {
            contact.RestoreFeasibility(fem.x, fem.dmask);
            break;
        }
        case EOgcTruncationStrategy::Global: {
            GloballyRestoreFeasibility(fem, contact, params, xt.derived());
            break;
        }
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.InitializeSolve");
    auto const xt = fem.bdf.CurrentState().reshaped(fem.x.rows(), fem.x.cols());
    contact.GetParams().ComputeQueryRadius((fem.xtilde - xt).colwise().norm().maxCoeff());
    contact.UpdateConstraintSet(xt);
    RestoreFeasibility(fem, contact, params, xt);
    params.newton.gk.resize(fem.x.size()); // Resize Newton optimizer's gradient buffer, because we
                                           // use it doubly for checking KKT conditions
    params.k = 0;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void LinearizeConstraints(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.LinearizeConstraints");
    contact.LinearizeConstraints(fem.x);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
bool CheckConvergence(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.CheckConvergence");
    ToGradient(fem, contact, params.newton.gk);
    params.newton.gknorm2 = params.newton.gk.squaredNorm();
    bool const bConverged = params.newton.gknorm2 <= params.newton.gtol2;
    return bConverged;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void PrepareSubproblem(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.PrepareSubproblem");
    ComputeElasticDerivatives(fem, contact, params);
    AssembleHessian(fem, contact, params, false /*bWithContacts*/);
    contact.UpdatePenaltyParameter(params.hessian);
    params.newton.InitializeSolve(fem.x);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void PrepareNextIteration(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params,
    bool bAreSubproblemDerivativesDirty)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.PrepareNextIteration");
    auto xk = fem.x.reshaped();
    params.newton.PrepareNextIteration(
        [&]([[maybe_unused]] auto const& _xk) {
            if (bAreSubproblemDerivativesDirty)
                ComputeElasticDerivatives<TElasticEnergy>(fem, contact, params);
            return MeritFunctionFromPrecomputedPotentials(fem, contact);
        } /* fPrepareDerivatives */,
        [&]([[maybe_unused]] auto const& _xk, Eigen::Vector<Scalar, Eigen::Dynamic>& gk) {
            ToGradient(fem, contact, gk);
        } /* g */,
        xk /* xk */);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
bool Iterate(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Iterate");
    auto xk = fem.x.reshaped();
    return params.newton.Iterate(
        [&]<class TDerivedX>(Eigen::MatrixBase<TDerivedX> const& xk) {
            return fem.Objective(xk) + contact.Potential(xk);
        } /* f */,
        [&]([[maybe_unused]] auto const& _xk,
            Eigen::Vector<Scalar, Eigen::Dynamic> const& gk,
            Eigen::Vector<Scalar, Eigen::Dynamic>& dxk) {
            AssembleHessian<TElasticEnergy>(fem, contact, params, true /*bWithContacts*/);
            HessianInverseProduct<TElasticEnergy>(gk, dxk, params);
        } /* Hinv */,
        xk /* xk */);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void FinalizeSubproblem(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.FinalizeSubproblem");
    using EDualVariable = MeshDynamics::EDualVariable;
    contact.UpdateDual<EDualVariable::Slack | EDualVariable::LagrangeMultiplier>(fem.x);
    contact.RestoreFeasibility(fem.x, fem.dmask);
    contact.UpdateConstraintSet(fem.x);
    ++params.k;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
bool Solve(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Solve");
    auto xk = fem.x.reshaped();
    for (; params.k < params.nMaxIters; ++params.k)
    {
        // 1. Linearize constraints
        contact.LinearizeConstraints(xk);
        // 2. Check KKT conditions and exit if converged
        ComputeElasticDerivatives(fem, contact, params);
        ComputeFrictionDerivatives(fem, contact);
        ToGradient(fem, contact, params.newton.gk);
        params.newton.gknorm2 = params.newton.gk.squaredNorm();
        if (params.newton.gknorm2 <= params.newton.gtol2)
            break;
        // 3. Update barrier parameters
        AssembleHessian(fem, contact, params, false /*bWithContacts*/);
        contact.UpdatePenaltyParameter(params.hessian);
        // 4. Newton solve the linear constraint subproblem
        params.newton.InitializeSolve(fem.x);
        [[maybe_unused]] bool const bSubproblemConverged = params.newton.Solve(
            [&]([[maybe_unused]] auto const& xk) {
                if (params.newton.k > 0)
                    ComputeElasticDerivatives<TElasticEnergy>(fem, contact, params);
                ComputeFrictionDerivatives(fem, contact);
                return MeritFunctionFromPrecomputedPotentials(fem, contact);
            } /* fPrepareDerivatives */,
            [&](auto const& xk) {
                Scalar Edyn = fem.Objective(xk);
                contact.ComputeEnergy(
                    xk,
                    -fem.bdf.Inertia(0),
                    fem.bdf.BetaTilde(),
                    MeshDynamics::EComputeFlags::Potential);
                Scalar Econ = contact.Potential(xk);
                return Edyn + Econ;
            } /* f */,
            [&]([[maybe_unused]] auto const& xk, Eigen::Vector<Scalar, Eigen::Dynamic>& gk) {
                ToGradient(fem, contact, gk);
            } /* g */,
            [&]([[maybe_unused]] auto const& _xk,
                Eigen::Vector<Scalar, Eigen::Dynamic> const& gk,
                Eigen::Vector<Scalar, Eigen::Dynamic>& dxk) {
                AssembleHessian<TElasticEnergy>(fem, contact, params, true /*bWithContacts*/);
                HessianInverseProduct<TElasticEnergy>(gk, dxk, params);
            } /* Hinv */,
            xk /* x0 */);
        // 5. Dual update
        using EDualVariable = typename MeshDynamics::EDualVariable;
        contact.UpdateDual<EDualVariable::Slack | EDualVariable::LagrangeMultiplier>(xk);
        // 5. Restore feasibility.
        GloballyRestoreFeasibility(fem, contact, params, contact.DynamicPointPositions());
        // 6. Update constraint set using the subproblem solution for ahead-of-time exploration.
        contact.UpdateConstraintSet(fem.x);
    }
    fem.BackSubstituteIntegratedPositionsIntoVelocities();
    bool const bConverged = params.newton.gknorm2 <= params.newton.gtol2;
    return bConverged;
}

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_CORE_H
