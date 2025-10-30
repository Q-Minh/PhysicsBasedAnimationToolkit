#ifndef PBAT_SIM_ALGORITHM_NEWTON_CORE_H
#define PBAT_SIM_ALGORITHM_NEWTON_CORE_H

#include "pbat/Aliases.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/math/optimization/Newton.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/dynamics/FemElastoDynamics.h"

#include <Eigen/Core>

namespace pbat::sim::algorithm::newton {

struct Params
{
    math::optimization::Newton<Scalar> newton;     ///< Newton optimizer
    Eigen::Vector<Index, Eigen::Dynamic> ordering; ///< Triplet ordering for sparse hessian assembly
    std::vector<Eigen::Triplet<Scalar, Index>> triplets; ///< Triplets for assembling the Hessian
    Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index> hessian; ///< Hessian matrix
};

/**
 * @brief Finite element elasto dynamics problem for VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
using FemElastoDynamics =
    dynamics::FemElastoDynamics<fem::Tetrahedron<1>, 3, TElasticEnergy, Scalar, Index>;

} // namespace pbat::sim::algorithm::newton

#endif // PBAT_SIM_ALGORITHM_NEWTON_CORE_H
