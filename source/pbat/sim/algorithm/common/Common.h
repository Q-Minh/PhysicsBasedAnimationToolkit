/**
 * @file Common.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Common utilities/definitions for simulation algorithms.
 * @version 0.1
 * @date 2025-10-28
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_COMMON_COMMON_H
#define PBAT_SIM_ALGORITHM_COMMON_COMMON_H

#include "pbat/fem/Tetrahedron.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/sim/dynamics/FemElastoDynamics.h"

/**
 * @brief Common utilities/definitions for simulation algorithms.
 * @namespace pbat::sim::algorithm::common
 */
namespace pbat::sim::algorithm::common {

/**
 * @brief Finite element elasto dynamics problem for VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
using FemElastoDynamics =
    dynamics::FemElastoDynamics<fem::Tetrahedron<1>, 3, TElasticEnergy, Scalar, Index>;

} // namespace pbat::sim::algorithm::common

#endif // PBAT_SIM_ALGORITHM_COMMON_COMMON_H
