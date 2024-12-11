#ifndef PBAT_SIM_VBD_MULTIGRID_MOMENTUM_ENERGY_H
#define PBAT_SIM_VBD_MULTIGRID_MOMENTUM_ENERGY_H

#include "Quadrature.h"
#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct MomentumEnergy
{
    MomentumEnergy() = default;
    /**
     * @brief
     * @param problem
     * @param CQ
     */
    MomentumEnergy(Data const& problem, CageQuadrature const& CQ);
    /**
     * @brief
     * @param xtilde
     */
    void UpdateInertialTargetPositions(Data const& problem);

    MatrixX xtildeg;  ///< 3x|#quad.pts.| array of inertial target positions at quad.pts.
    IndexVectorX erg; ///< |#quad.pts.| array of root mesh elements associated with quad.pts.
    MatrixX Nrg;      ///< 3x|#quad.pts.| array of root mesh shape functions at quad.pts.
    VectorX rhog;     ///< |#quad.pts.| array of mass densities at quad.pts.
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_MOMENTUM_ENERGY_H