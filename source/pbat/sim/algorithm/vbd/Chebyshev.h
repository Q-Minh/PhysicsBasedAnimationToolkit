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

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Chebyshev accelerated VBD solver parameters
 */
struct ChebyshevParams
{
    Scalar rho; ///< Spectral radius estimate
    /**
     * @brief Reset transient data to start a new solve
     */
    void ResetTransientData() { k = 0; }
    /**
     * @brief Read/Write parameters
     */
    Index k{0};   ///< Current iteration
    Scalar omega; ///< Chebyshev omega parameter
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
        xkm1; ///< `3 x |# verts|` \f$ x^{k-1} \f$ used in Chebyshev semi-iterative method
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic>
        xkm2; ///< `3 x |# verts|` \f$ x^{k-2} \f$ used in Chebyshev semi-iterative method
};

/**
 * @brief One Chebyshev accelerated VBD minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param cheb Chebyshev parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Step(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, ChebyshevParams& cheb);

template <physics::CHyperElasticEnergy TElasticEnergy>
void Step(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, ChebyshevParams& cheb)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Chebyshev.Step");
    Scalar rho2 = cheb.rho * cheb.rho;
    Step(fem, params);
    // Chebyshev Update
    cheb.omega = kernels::ChebyshevOmega(cheb.k, rho2, cheb.omega);
    auto& xk   = fem.x;
    if (cheb.k > 1)
        xk = cheb.omega * (xk - cheb.xkm2) + cheb.xkm2;
    cheb.xkm2 = cheb.xkm1;
    cheb.xkm1 = xk;
}

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_CHEBYSHEV_H
