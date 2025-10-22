/**
 * @file Chebyshev.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Chebyshev accelerated VBD.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_CHEBYSHEV_H
#define PBAT_SIM_ALGORITHM_VBD_CHEBYSHEV_H

#include "Core.h"
#include "pbat/io/Archive.h"
#include "pbat/profiling/Profiling.h"

#include <optional>

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Chebyshev accelerated VBD solver parameters
 * @details See @cite wang_chebyshev_2015, @cite anka2024vbd
 */
struct ChebyshevParams
{
    Scalar rho{0.9}; ///< Spectral radius estimate `0 < \rho < 1`
    /**
     * @brief Read/Write parameters
     */
    Index k;      ///< Iteration
    Scalar rho2;  ///< Square of spectral radius estimate
    Scalar omega; ///< Chebyshev omega parameter
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
        xkm1; ///< `3 x |# verts|` \f$ x^{k-1} \f$ used in Chebyshev semi-iterative method
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
        xkm2; ///< `3 x |# verts|` \f$ x^{k-2} \f$ used in Chebyshev semi-iterative method

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
     * @brief Allocate memory for Chebyshev parameters if necessary
     * @param n Number of vertices
     */
    void AllocateIfNeeded(Index n)
    {
        xkm1.resize(3, n);
        xkm2.resize(3, n);
    }
};

/**
 * @brief Initialize Chebyshev accelerated VBD minimization solve
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param cheb Chebyshev parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    ChebyshevParams& cheb);

/**
 * @brief One Chebyshev accelerated VBD minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param cheb Chebyshev parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, ChebyshevParams& cheb);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using
 * Chebyshev-accelerated VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param cheb Chebyshev parameters
 * @param ac Optional archive to serialize to
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    ChebyshevParams& cheb,
    std::optional<io::Archive> ac = std::nullopt);

/**
 * @brief Integrate FEM elasto dynamics one step using Chebyshev-accelerated VBD as the non-linear
 * solver
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param cheb Chebyshev parameters
 * @param ac Optional archive to serialize to
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    ChebyshevParams& cheb,
    std::optional<io::Archive> ac = std::nullopt);

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    ChebyshevParams& cheb)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Chebyshev.InitializeSolve");
    cheb.AllocateIfNeeded(fem.x.cols());
    InitializeSolve<TElasticEnergy>(fem, params);
    cheb.k    = 0;
    cheb.rho2 = cheb.rho * cheb.rho;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, ChebyshevParams& cheb)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Chebyshev.Iterate");
    Iterate(fem, params);
    // Chebyshev Update
    cheb.omega = kernels::ChebyshevOmega(cheb.k, cheb.rho2, cheb.omega);
    auto& xk   = fem.x;
    if (cheb.k > 1)
        xk = cheb.omega * (xk - cheb.xkm2) + cheb.xkm2;
    cheb.xkm2 = cheb.xkm1;
    cheb.xkm1 = xk;
    ++cheb.k;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    ChebyshevParams& cheb,
    std::optional<io::Archive> ac)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Chebyshev.Solve");
    std::optional<io::Archive> group;
    if (ac)
    {
        group = ac->GetOrCreateGroup("pbat.sim.algorithm.vbd.Chebyshev.Solve");
    }
    InitializeSolve<TElasticEnergy>(fem, params, cheb);
    for (; cheb.k < params.nMaxIters;)
    {
        if (group)
        {
            SerializeSolverIteration(fem, cheb.k, *group);
        }
        Iterate<TElasticEnergy>(fem, params, cheb);
    }
    BackSubstituteIntegratedPositionsIntoVelocities<TElasticEnergy>(fem, params);
    if (group)
    {
        SerializeSolverIteration(fem, cheb.k, *group, true /* bPostSolve */);
    }
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    ChebyshevParams& cheb,
    std::optional<io::Archive> ac)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Chebyshev.Integrate");
    fem.SetupTimeIntegrationOptimization();
    std::optional<io::Archive> group;
    if (ac)
    {
        group = ac->GetOrCreateGroup("pbat.sim.algorithm.vbd.Chebyshev.Integrate");
        fem.Serialize(*group);
    }
    Solve<TElasticEnergy>(fem, params, cheb, group);
    fem.Step();
}

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_CHEBYSHEV_H
