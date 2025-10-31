#ifndef PBAT_SIM_ALGORITHM_NEWTON_CORE_H
#define PBAT_SIM_ALGORITHM_NEWTON_CORE_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/io/Archive.h"
#include "pbat/math/optimization/Newton.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/dynamics/FemElastoDynamics.h"

#ifdef PBAT_USE_SUITESPARSE
    #include "pbat/math/linalg/CholmodSupport.h"
#else
    #include <Eigen/SparseCholesky>
#endif // PBAT_USE_SUITESPARSE

#include <Eigen/Core>
#include <algorithm>
#include <exception>
#include <fmt/core.h>

namespace pbat::sim::algorithm::newton {

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
        eSpdCorrection; ///< SPD correction method for hyper-elastic Hessians

#ifdef PBAT_USE_SUITESPARSE
    using DecompositionType =
        Eigen::CholmodDecomposition<decltype(hessian), Eigen::Lower>; ///< Cholesky decomposition
                                                                      ///< type
#else
    using DecompositionType =
        Eigen::SimplicialLDLT<decltype(hessian)>; ///< Cholesky decomposition type
#endif                     // PBAT_USE_SUITESPARSE
    DecompositionType llt; ///< Cholesky decomposition of the Hessian
};

/**
 * @brief Finite element elasto dynamics problem for VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
using FemElastoDynamics =
    dynamics::FemElastoDynamics<fem::Tetrahedron<1>, 3, TElasticEnergy, Scalar, Index>;

/**
 * @brief Prepare next iteration for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void PrepareNextIteration(FemElastoDynamics<TElasticEnergy>& fem, Params& params);

/**
 * @brief Initialize the solve process for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(FemElastoDynamics<TElasticEnergy>& fem, Params& params);

/**
 * @brief One Newton minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @return true if step was taken, false otherwise
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
bool Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using Newton's method
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @return true if converged, false otherwise
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
bool Solve(FemElastoDynamics<TElasticEnergy>& fem, Params const& params);

/**
 * @brief Derivative precomputation for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
Scalar PrepareDerivatives(FemElastoDynamics<TElasticEnergy>& fem, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.PrepareDerivatives");
    // Precompute elastic energy and its derivatives
    fem.ComputeElasticEnergy(
        fem::EElementElasticityComputationFlags::Potential |
            fem::EElementElasticityComputationFlags::Gradient |
            fem::EElementElasticityComputationFlags::Hessian,
        params.eSpdCorrection);
    Scalar bt  = fem.bdf.BetaTilde();
    Scalar bt2 = bt * bt;
    fem.HgU *= bt2;
    fem.GgU *= bt2;
    Scalar U = fem::HyperElasticPotential(fem.UgU);
    Scalar K = fem.DiscreteKineticEnergy();
    return K + bt * bt * U /* + C*/;
}

/**
 * @brief Derivative precomputation for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param gk Gradient vector
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void ToGradient(
    FemElastoDynamics<TElasticEnergy> const& fem,
    Eigen::Vector<Scalar, Eigen::Dynamic>& gk)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.ToGradient");
    // Gradient of 1/2 |x - \Tilde{x}|_M^2 + bt^2 U(x)
    fem::ToHyperElasticGradient(fem.mesh, fem.egU, fem.GgU, gk);
    gk += ((fem.x - fem.xtilde) * fem.m.asDiagonal()).reshaped();
    gk(fem.DirichletDofs()).setZero();
}

/**
 * @brief Compute objective function for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @return Objective function value
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
Scalar ObjectiveFunction(FemElastoDynamics<TElasticEnergy>& fem)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.ObjectiveFunction");
    // Objective function 1/2 |x - \Tilde{x}|_M^2 + bt^2 U(x)
    return fem.Objective() /* + C */;
}

/**
 * @brief Assemble the Hessian for the given finite element elasto dynamics problem.
 *
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem
 * @param params Solver parameters
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void AssembleHessian(FemElastoDynamics<TElasticEnergy> const& fem, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.AssembleHessian");
    // Hessian of 1/2 |x - \Tilde{x}|_M^2 + bt^2 U(x)
    auto nTriplets = fem.HgU.size() + fem.M().size() /* + |C|*/;
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
        auto HUg = fem.HgU.template block<kDims * ElementType::kNodes, kDims * ElementType::kNodes>(
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
    // Assemble
    params.hessian.resize(fem.x.size(), fem.x.size());
    // Remove off-diagonal Dirichlet entries and upper triangular part
    auto itRemoveBegin = std::remove_if(
        params.triplets.begin(),
        params.triplets.end(),
        [&](Eigen::Triplet<Scalar, Index> const& triplet) {
            bool bIsUpperTriangular = triplet.row() < triplet.col();
            bool bIsDiag            = triplet.row() == triplet.col();
            bool bIsDirichletEntry =
                fem.IsDirichletDof(triplet.row()) or fem.IsDirichletDof(triplet.col());
            return bIsUpperTriangular or (not bIsDiag and bIsDirichletEntry);
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
    params.llt.compute(params.hessian);
    if (params.llt.info() != Eigen::Success)
    {
        throw std::runtime_error(
            fmt::format(
                "Cholesky decomposition failed with info code {}",
                static_cast<int>(params.llt.info())));
    }
    dxk = params.llt.solve(gk);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void PrepareNextIteration(FemElastoDynamics<TElasticEnergy>& fem, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.PrepareNextIteration");
    auto xk = fem.x.reshaped();
    params.newton.PrepareNextIteration(
        [&]([[maybe_unused]] auto const& xk) {
            return PrepareDerivatives<TElasticEnergy>(fem, params);
        } /* fPrepareDerivatives */,
        [&]([[maybe_unused]] auto const& xk, Eigen::Vector<Scalar, Eigen::Dynamic>& gk) {
            ToGradient<TElasticEnergy>(fem, gk);
        } /* g */,
        xk /* xk */);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(FemElastoDynamics<TElasticEnergy>& fem, Params& params)
{
    PrepareNextIteration(fem, params);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
bool Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Iterate");
    auto xk = fem.x.reshaped();
    return params.newton.Iterate(
        [&]([[maybe_unused]] auto const& xk) {
            return ObjectiveFunction<TElasticEnergy>(fem);
        } /* f */,
        [&]([[maybe_unused]] auto const& xk,
            Eigen::Vector<Scalar, Eigen::Dynamic> const& gk,
            Eigen::Vector<Scalar, Eigen::Dynamic>& dxk) {
            AssembleHessian<TElasticEnergy>(fem, params);
            HessianInverseProduct<TElasticEnergy>(gk, dxk, params);
        } /* Hinv */,
        xk /* xk */);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
bool Solve(FemElastoDynamics<TElasticEnergy>& fem, Params& params)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Solve");
    auto x0 = fem.x.reshaped();
    return params.newton.Solve(
        [&]([[maybe_unused]] auto const& xk) {
            return PrepareDerivatives<TElasticEnergy>(fem, params);
        } /* fPrepareDerivatives */,
        [&]([[maybe_unused]] auto const& xk) {
            return ObjectiveFunction<TElasticEnergy>(fem);
        } /* f */,
        [&]([[maybe_unused]] auto const& xk, Eigen::Vector<Scalar, Eigen::Dynamic>& gk) {
            ToGradient<TElasticEnergy>(fem, gk);
        } /* g */,
        [&]([[maybe_unused]] auto const& xk,
            Eigen::Vector<Scalar, Eigen::Dynamic> const& gk,
            Eigen::Vector<Scalar, Eigen::Dynamic>& dxk) {
            AssembleHessian<TElasticEnergy>(fem, params);
            HessianInverseProduct<TElasticEnergy>(gk, dxk, params);
        } /* Hinv */,
        x0 /* xk */);
}

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_CORE_H
