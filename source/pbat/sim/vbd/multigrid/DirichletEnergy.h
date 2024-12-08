#ifndef PBAT_SIM_VBD_MULTIGRID_DIRICHLET_ENERGY_H
#define PBAT_SIM_VBD_MULTIGRID_DIRICHLET_ENERGY_H

#include "Mesh.h"
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
     * @param CM
     * @param DQ
     */
    DirichletEnergy(Data const& problem, VolumeMesh const& CM, DirichletQuadrature const& DQ);

    Scalar muD{1}; ///< Dirichlet penalty coefficient
    MatrixX Ncg;   ///< 4x|#quad.pts.| matrix of coarse mesh shape functions at Dirichlet quad.pts.
    MatrixX dg;    ///< 3x|#quad.pts.| Dirichlet conditions
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_DIRICHLET_ENERGY_H