#ifndef PBAT_SIM_VBD_LOD_ELASTIC_ENERGY_H
#define PBAT_SIM_VBD_LOD_ELASTIC_ENERGY_H

#include "Quadrature.h"
#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

struct ElasticEnergy
{
    ElasticEnergy() = default;
    /**
     * @brief
     * @param problem
     * @param CQ
     */
    ElasticEnergy(Data const& problem, CageQuadrature const& CQ);

    VectorX mug;     ///< |#quad.pts.| array of first Lame coefficients at quadrature points
    VectorX lambdag; ///< |#quad.pts.| array of second Lame coefficients at quadrature points
};

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LOD_ELASTIC_ENERGY_H