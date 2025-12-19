#ifndef PBAT_SIM_ALGORITHM_NEWTON_CORE_H
#define PBAT_SIM_ALGORITHM_NEWTON_CORE_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/io/Archive.h"
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
#include <fmt/core.h>
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

    math::optimization::Newton<Scalar> newton;     ///< Newton optimizer
    Eigen::Vector<Index, Eigen::Dynamic> ordering; ///< Triplet ordering for sparse hessian assembly
    std::vector<Eigen::Triplet<Scalar, Index>> triplets; ///< Triplets for assembling the Hessian
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index> hessian; ///< Hessian matrix
    fem::EHyperElasticSpdCorrection
        eSpdCorrection;          ///< SPD correction method for hyper-elastic Hessians
    ELinearSolver eLinearSolver; ///< Linear solver type for Newton step

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
    SolverType Hinv; ///< Hessian inverse
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
 * @brief Prepare next iteration for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void PrepareNextIteration(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params);

/**
 * @brief Initialize the solve process for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params);

/**
 * @brief One Newton minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param contact Mesh contact dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @return true if step was taken, false otherwise
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
bool Iterate(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params const& params);

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
bool Solve(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params const& params);

/**
 * @brief Truncate Newton step for OGC.
 * @tparam TDerivedxk Eigen matrix type for xt
 * @tparam TDerivedxkp1 Eigen matrix type for x
 * @param contact The mesh contact dynamics problem
 * @param xk `3 x |# points|` or `3*|# points| x 1` matrix of previous positions
 * @param xkp1 `3 x |# points|` or `3*|# points| x 1` matrix of current positions
 * @return (min,max) step size after truncation
 */
template <class TDerivedxk, class TDerivedxkp1>
std::pair<Scalar, Scalar> MinMaxStepSize(
    MeshDynamics const& contact,
    Eigen::MatrixBase<TDerivedxk> const& _xk,
    Eigen::MatrixBase<TDerivedxkp1> const& _xkp1)
{
    auto xkp1          = _xkp1.derived().reshaped(3, _xkp1.size() / 3);
    auto xk            = _xk.derived().reshaped(3, _xk.size() / 3);
    auto const dnorms  = (xkp1 - xk).colwise().norm();
    Scalar const dnorm = dnorms.maxCoeff();
    Scalar const dmin  = contact.OgcState().bv.minCoeff();
    return {dmin, dnorm};
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
Scalar
PrepareDerivatives(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.PrepareDerivatives");
    // Precompute elastic energy and its derivatives
    fem.ComputeElasticEnergy(
        fem.x,
        fem::EElementElasticityComputationFlags::Potential |
            fem::EElementElasticityComputationFlags::Gradient |
            fem::EElementElasticityComputationFlags::Hessian,
        params.eSpdCorrection);
    contact.ComputeEnergies(
        fem.x,
        sim::contact::EMeshEnergyComputationFlags::Potential |
            sim::contact::EMeshEnergyComputationFlags::Gradient |
            sim::contact::EMeshEnergyComputationFlags::Hessian);
    Scalar bt  = fem.bdf.BetaTilde();
    Scalar bt2 = bt * bt;
    fem.HgU *= bt2;
    fem.GgU *= bt2;
    contact.ForEachMeshContactEnergy(
        [&]<int kStencil>(sim::contact::MeshContactEnergy<Scalar, Index, kStencil>& energy) {
            energy.gradEn *= bt2;
            energy.gradEf *= bt2;
            energy.hessEn *= bt2;
            energy.hessEf *= bt2;
        });
    Scalar U = fem::HyperElasticPotential(fem.UgU);
    Scalar K = fem.DiscreteKineticEnergy(fem.x);
    Scalar C = contact.Potential();
    return K + bt2 * U + bt2 * C;
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
    MeshDynamics const& contact,
    Eigen::Vector<Scalar, Eigen::Dynamic>& gk)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.ToGradient");
    // Gradient of 1/2 |x - \Tilde{x}|_M^2 + bt^2 U(x) + bt^2 C(x)
    fem::ToHyperElasticGradient(fem.mesh, fem.egU, fem.GgU, gk);
    contact.ToGradient(gk);
    gk += ((fem.x - fem.xtilde) * fem.m.asDiagonal()).reshaped();
    gk(fem.DirichletDofs()).setZero();
}

/**
 * @brief Assemble the Hessian for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param contact Mesh contact problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void AssembleHessian(
    FemElastoDynamics<TElasticEnergy> const& fem,
    MeshDynamics& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.AssembleHessian");
    // Hessian of 1/2 |x - \Tilde{x}|_M^2 + bt^2 U(x) + bt^2 C(x)
    auto constexpr kVertexVertexStencil   = 2;
    auto constexpr kVertexEdgeStencil     = 3;
    auto constexpr kVertexTriangleStencil = 4;
    auto constexpr kEdgeEdgeStencil       = 4;
    auto constexpr kVertexEnvStencil      = 1;
    auto constexpr kEdgeEnvStencil        = 2;
    auto constexpr kTriangleEnvStencil    = 3;
    auto const nTriplets =
        fem.HgU.size() + fem.M().size() +
        contact.NumVertexVertexContacts() * 9 * kVertexVertexStencil * kVertexVertexStencil +
        contact.NumVertexEdgeContacts() * 9 * kVertexEdgeStencil * kVertexEdgeStencil +
        contact.NumVertexTriangleContacts() * 9 * kVertexTriangleStencil * kVertexTriangleStencil +
        contact.NumEdgeEdgeContacts() * 9 * kEdgeEdgeStencil * kEdgeEdgeStencil +
        contact.NumVertexEnvironmentContacts() * 9 * kVertexEnvStencil * kVertexEnvStencil +
        contact.NumEdgeEnvironmentContacts() * 9 * kEdgeEnvStencil * kEdgeEnvStencil +
        contact.NumTriangleEnvironmentContacts() * 9 * kTriangleEnvStencil * kTriangleEnvStencil;
    params.triplets.resize(nTriplets);
    // Assemble
    std::size_t k{0};
    auto constexpr kDims = std::decay_t<decltype(fem)>::kDims;
    // Mass matrix contribution
    for (Eigen::Index i = 0; i < fem.m.size(); ++i)
        for (auto d = 0; d < kDims; ++d)
            params.triplets[k++] =
                Eigen::Triplet<Scalar, Index>(i * kDims + d, i * kDims + d, fem.m(i));
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
                        params.triplets[k++] = Eigen::Triplet<Scalar, Index>(
                            nodes(il) * kDims + id,
                            nodes(jl) * kDims + jd,
                            HUg(il * kDims + id, jl * kDims + jd));
    }
    // Contact Hessian contribution
    contact.ForEachMeshContactEnergy(
        [&]<int kStencil>(sim::contact::MeshContactEnergy<Scalar, Index, kStencil> const& E) {
            for (auto jl = 0; jl < kStencil; ++jl)
                for (auto jd = 0; jd < kDims; ++jd)
                    for (auto il = 0; il < kStencil; ++il)
                        for (auto id = 0; id < kDims; ++id)
                            params.triplets[k++] = Eigen::Triplet<Scalar, Index>(
                                E.stencil(il) * kDims + id,
                                E.stencil(jl) * kDims + jd,
                                E.hessEn(il * kDims + id, jl * kDims + jd) +
                                    E.hessEf(il * kDims + id, jl * kDims + jd));
        });
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
            return (bIsUpperTriangular) or (not bIsDiag and bIsDirichletEntry);
        });
    params.triplets.erase(itRemoveBegin, params.triplets.end());
    params.hessian.setFromTriplets(params.triplets.begin(), params.triplets.end());
    // Make sure the hessian is symmetric after Eigen's setFromTriplets, which is subject to
    // rounding errors
    if (params.eLinearSolver != ELinearSolver::LLT)
    {
        using SparseMatrixType = decltype(params.hessian);
        params.hessian         = (Scalar(0.5) * params.hessian) +
                         SparseMatrixType(Scalar(0.5) * params.hessian.transpose());
    }
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

template <physics::CHyperElasticEnergy TElasticEnergy>
void PrepareNextIteration(
    FemElastoDynamics<TElasticEnergy>& fem,
    MeshDynamics& contact,
    Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.PrepareNextIteration");
    auto xk = fem.x.reshaped();
    params.newton.PrepareNextIteration(
        [&]([[maybe_unused]] auto const& _xk) {
            if (contact.RequiresBoundsComputation())
                contact.ComputeDisplacementBounds(fem.x);
            return PrepareDerivatives<TElasticEnergy>(fem, contact, params);
        } /* fPrepareDerivatives */,
        [&]([[maybe_unused]] auto const& _xk, Eigen::Vector<Scalar, Eigen::Dynamic>& gk) {
            ToGradient<TElasticEnergy>(fem, contact, gk);
        } /* g */,
        xk /* xk */);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params)
{
    params.newton.k = 0;
    auto const xt   = fem.bdf.CurrentState().reshaped(fem.x.rows(), fem.x.cols());
    auto& ogcParams = contact.GetParams().mOgcParams;
    ogcParams.rq    = ogcParams.r + (fem.xtilde - xt).colwise().norm().maxCoeff();
    contact.ComputeDisplacementBounds(xt);
    Scalar const dmax = (fem.x - xt).colwise().norm().maxCoeff();
    Scalar const dmin = contact.OgcState().bv.minCoeff();
    if (dmax > dmin)
    {
        fem.x(Eigen::placeholders::all, fem.FreeNodes()) =
            xt(Eigen::placeholders::all, fem.FreeNodes()) +
            (dmin / dmax) * (fem.x - xt)(Eigen::placeholders::all, fem.FreeNodes());
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
bool Iterate(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Iterate");
    auto xk = fem.x.reshaped();
    return params.newton.Iterate(
        [&]<class TDerivedX>(Eigen::MatrixBase<TDerivedX> const& xk) {
            contact.ComputeEnergies(xk, sim::contact::EMeshEnergyComputationFlags::Potential);
            auto const bt  = fem.bdf.BetaTilde();
            auto const bt2 = bt * bt;
            return fem.Objective(xk) + bt2 * contact.Potential();
        } /* f */,
        [&]([[maybe_unused]] auto const& _xk,
            Eigen::Vector<Scalar, Eigen::Dynamic> const& gk,
            Eigen::Vector<Scalar, Eigen::Dynamic>& dxk) {
            AssembleHessian<TElasticEnergy>(fem, contact, params);
            HessianInverseProduct<TElasticEnergy>(gk, dxk, params);
            Scalar dmax = dxk.reshaped(fem.x.rows(), fem.x.cols()).colwise().norm().maxCoeff();
            Scalar dmin = contact.OgcState().bv.minCoeff();
            if (dmax > dmin)
            {
                dxk *= (dmin / dmax);
                contact.RequestDisplacementBoundsComputation();
            }
        } /* Hinv */,
        xk /* xk */);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
bool Solve(FemElastoDynamics<TElasticEnergy>& fem, MeshDynamics& contact, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Solve");
    auto x0 = fem.x.reshaped();
    return params.newton.Solve(
        [&]([[maybe_unused]] auto const& xk) {
            if (contact.RequiresBoundsComputation())
                contact.ComputeDisplacementBounds(fem.x);
            return PrepareDerivatives<TElasticEnergy>(fem, contact, params);
        } /* fPrepareDerivatives */,
        [&]<class TDerivedX>(Eigen::MatrixBase<TDerivedX> const& xk) {
            contact.ComputeEnergies(xk, sim::contact::EMeshEnergyComputationFlags::Potential);
            auto const bt  = fem.bdf.BetaTilde();
            auto const bt2 = bt * bt;
            return fem.Objective(xk) + bt2 * contact.Potential();
        } /* f */,
        [&]([[maybe_unused]] auto const& xk, Eigen::Vector<Scalar, Eigen::Dynamic>& gk) {
            ToGradient<TElasticEnergy>(fem, contact, gk);
        } /* g */,
        [&]([[maybe_unused]] auto const& _xk,
            Eigen::Vector<Scalar, Eigen::Dynamic> const& gk,
            Eigen::Vector<Scalar, Eigen::Dynamic>& dxk) {
            AssembleHessian<TElasticEnergy>(fem, contact, params);
            HessianInverseProduct<TElasticEnergy>(gk, dxk, params);
            Scalar dmax = dxk.reshaped(fem.x.rows(), fem.x.cols()).colwise().norm().maxCoeff();
            Scalar dmin = contact.OgcState().bv.minCoeff();
            if (dmax > dmin)
            {
                dxk *= (dmin / dmax);
                contact.RequestDisplacementBoundsComputation();
            }
        } /* Hinv */,
        x0 /* xk */);
}

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_CORE_H
