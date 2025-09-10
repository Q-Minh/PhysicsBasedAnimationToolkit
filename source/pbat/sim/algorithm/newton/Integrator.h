/**
 * @file Integrator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for the Newton integrator.
 * @date 2025-05-05
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_ALGORITHM_NEWTON_INTEGRATOR_H
#define PBAT_SIM_ALGORITHM_NEWTON_INTEGRATOR_H

#include "Config.h"
#include "HessianProduct.h"
#include "Preconditioner.h"
#include "pbat/Aliases.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/io/Archive.h"
#include "pbat/math/optimization/LineSearch.h"
#include "pbat/math/optimization/Newton.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"
#include "pbat/sim/dynamics/FemElastoDynamics.h"

#include <optional>

namespace pbat::sim::algorithm::newton {

/**
 * @brief Newton dynamics integrator.
 */
class Integrator
{
  public:
    static auto constexpr kDims = 3; ///< Number of spatial dimensions
    static auto constexpr kOrder = 1; ///< Shape function order
    using ElastoDynamicsType    = dynamics::FemElastoDynamics<
           fem::Tetrahedron<1>,
           kDims,
           physics::StableNeoHookeanEnergy<kDims>>; ///< Elasto-dynamics type

    Integrator(Config config, ElastoDynamicsType elastoDynamics);
    Integrator(Integrator const&)            = delete;
    Integrator(Integrator&&)                 = default;
    Integrator& operator=(Integrator const&) = delete;
    Integrator& operator=(Integrator&&)      = default;

    void Step(std::optional<io::Archive> archive = std::nullopt);

    Config mConfig;                     ///< Configuration for the Newton integrator
    ElastoDynamicsType mElastoDynamics; ///< Hyper elasticity dynamics

  private:
    math::optimization::Newton<Scalar> mNewton;                     ///< Newton optimization solver
    math::optimization::BackTrackingLineSearch<Scalar> mLineSearch; ///< Line searcher
    Hessian mHessian;                                               ///< Hessian storage
    VectorX mGradU;                 ///< Gradient of the elastic potential
    Preconditioner mPreconditioner; ///< Preconditioner
};

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_INTEGRATOR_H
