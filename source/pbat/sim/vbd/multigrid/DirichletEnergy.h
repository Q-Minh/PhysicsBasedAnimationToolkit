#ifndef PBAT_SIM_VBD_MULTIGRID_DIRICHLET_ENERGY_H
#define PBAT_SIM_VBD_MULTIGRID_DIRICHLET_ENERGY_H

#include "Quadrature.h"
#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct DirichletEnergy
{
    DirichletEnergy() = default;
    /**
     * @brief
     * @param problem
     * @param DQ
     */
    DirichletEnergy(Data const& problem, DirichletQuadrature const& DQ);

    Scalar muD{1}; ///< Dirichlet penalty coefficient
    MatrixX dg;    ///< 3x|#quad.pts.| Dirichlet conditions
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_DIRICHLET_ENERGY_H