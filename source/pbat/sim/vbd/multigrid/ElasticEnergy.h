#ifndef PBAT_SIM_VBD_MULTIGRID_ELASTIC_ENERGY_H
#define PBAT_SIM_VBD_MULTIGRID_ELASTIC_ENERGY_H

#include "Mesh.h"
#include "Quadrature.h"
#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct ElasticEnergy
{
    ElasticEnergy() = default;
    /**
     * @brief
     * @param problem
     * @param CM
     * @param CQ
     */
    ElasticEnergy(Data const& problem, VolumeMesh const& CM, CageQuadrature const& CQ);

    VectorX mug;     ///< |#quad.pts.| array of first Lame coefficients at quadrature points
    VectorX lambdag; ///< |#quad.pts.| array of second Lame coefficients at quadrature points
    MatrixX GNg;     ///< 4x|3*#quad.pts.| array of coarse cage element shape function gradients at
                     ///< quadrature points
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_ELASTIC_ENERGY_H