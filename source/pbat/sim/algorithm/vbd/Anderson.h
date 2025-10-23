/**
 * @file Anderson.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Anderson accelerated for VBD.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_ANDERSON_H
#define PBAT_SIM_ALGORITHM_VBD_ANDERSON_H

#include "Core.h"
#include "pbat/common/Modulo.h"
#include "pbat/io/Archive.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/QR>
#include <exception>
#include <fmt/core.h>
#include <optional>

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Anderson accelerated VBD solver parameters
 *
 * @details See @cite anderson_iterative_1965, @cite fang_two_2009
 */
struct AndersonParams
{
    Index m{5};                     ///< Window size
    Scalar beta{1};                 ///< Mixing parameter
    Scalar codNumericalZero{1e-10}; ///< Numerical zero threshold for COD solver
    /**
     * @brief Read/Write parameters
     */
    Index k;        ///< Current iteration
    MatrixX Fk;     ///< `|# dofs| x m` residual differences
    MatrixX Xk;     ///< `|# dofs| x m` past step differences
    VectorX xkm1;   ///< `|# dofs| x 1` previous step
    VectorX fk;     ///< `|# dofs| x 1` current residual
    VectorX fkm1;   ///< `|# dofs| x 1` past residual
    VectorX gammak; ///< `m x 1` subspace residual
    /**
     * @brief Least-squares solver
     */
    Eigen::CompleteOrthogonalDecomposition<MatrixX> cod; ///< COD solver for least-squares problem

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

    /**
     * @brief Allocate memory for Anderson parameters
     * @param n Number of degrees of freedom
     */
    void AllocateIfNeeded(Index n);
};

/**
 * @brief Initialize Anderson accelerated VBD minimization solve
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param anderson Anderson parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    AndersonParams& anderson);

/**
 * @brief One Anderson accelerated VBD minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param anderson Anderson parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    AndersonParams& anderson);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using Anderson-accelerated
 * VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param anderson Anderson parameters
 * @param ac Optional archive to serialize to
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    AndersonParams& anderson,
    std::optional<io::Archive> ac = std::nullopt);

/**
 * @brief Integrate FEM elasto dynamics one step using Anderson-accelerated VBD as the non-linear
 * solver
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param anderson Anderson parameters
 * @param ac Optional archive to serialize to
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    AndersonParams& anderson,
    std::optional<io::Archive> ac = std::nullopt);

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    AndersonParams& anderson)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Anderson.InitializeSolve");
    InitializeSolve<TElasticEnergy>(fem, params);
    anderson.AllocateIfNeeded(fem.x.size());
    anderson.xkm1 = fem.x.reshaped();
    Iterate(fem, params);
    anderson.fkm1 = fem.x.reshaped() - anderson.xkm1;
    anderson.cod.setThreshold(anderson.codNumericalZero);
    anderson.k = 1;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, AndersonParams& anderson)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Anderson.Iterate");
    auto dkl             = common::Modulo(anderson.k - 1, anderson.m);
    anderson.Xk.col(dkl) = fem.x.reshaped() - anderson.xkm1;
    anderson.xkm1        = fem.x.reshaped();
    Iterate(fem, params);
    anderson.fk          = fem.x.reshaped() - anderson.xkm1;
    anderson.Fk.col(dkl) = anderson.fk - anderson.fkm1;
    anderson.fkm1        = anderson.fk;
    auto mk              = std::min(anderson.m, anderson.k);
    auto Fk              = anderson.Fk.leftCols(mk);
    // NOTE: I would like to use a COD or QR updating scheme here instead of recomputing from
    // scratch every time (Eigen does not seem to support it), but the updating scheme needs to
    // account for pivoting as well.
    anderson.cod.compute(Fk);
    if (anderson.cod.info() != Eigen::ComputationInfo::Success)
    {
        throw std::runtime_error(
            fmt::format("COD decomposition failed at iteration {}", anderson.k));
    }
    anderson.gammak.head(mk) = anderson.cod.solve(anderson.fk);
    // At this point, anderson.xkm1 contains x_k, while fem.x.reshaped() contains x_k + f_k
    fem.x.reshaped() = anderson.xkm1 + anderson.beta * anderson.fk;
    fem.x.reshaped() -= anderson.Xk.leftCols(mk) * anderson.gammak.head(mk);
    fem.x.reshaped() -= anderson.beta * (anderson.Fk.leftCols(mk) * anderson.gammak.head(mk));
    ++anderson.k;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    AndersonParams& anderson,
    std::optional<io::Archive> ac)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Anderson.Solve");
    InitializeSolve<TElasticEnergy>(fem, params, anderson);
    std::optional<io::Archive> group;
    if (ac)
    {
        group = ac->GetOrCreateGroup("pbat.sim.algorithm.vbd.Anderson.Solve");
        SerializeSolverIteration<TElasticEnergy>(fem, 0, *group);
    }
    for (; anderson.k < params.nMaxIters;)
    {
        if (group)
        {
            SerializeSolverIteration<TElasticEnergy>(fem, anderson.k, *group);
        }
        Iterate<TElasticEnergy>(fem, params, anderson);
    }
    BackSubstituteIntegratedPositionsIntoVelocities<TElasticEnergy>(fem, params);
    if (group)
    {
        SerializeSolverIteration<TElasticEnergy>(fem, anderson.k, *group, true /* bPostSolve */);
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    AndersonParams& anderson,
    std::optional<io::Archive> ac)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Anderson.Integrate");
    fem.SetupTimeIntegrationOptimization();
    std::optional<io::Archive> group;
    if (ac)
    {
        group = ac->GetOrCreateGroup("pbat.sim.algorithm.vbd.Anderson.Integrate");
        fem.Serialize(*group);
    }
    Solve<TElasticEnergy>(fem, params, anderson, group);
    fem.Step();
}

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_ANDERSON_H
