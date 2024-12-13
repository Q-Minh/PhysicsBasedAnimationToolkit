#ifndef PBAT_SIM_VBD_MULTIGRID_LEVEL_H
#define PBAT_SIM_VBD_MULTIGRID_LEVEL_H

#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Level
{
    MatrixX u;             ///< 3x|#cage verts| coarse displacement coefficients
    IndexVectorX colors;   ///< Coarse vertex graph coloring
    IndexVectorX ptr, adj; ///< Parallel vertex partitions
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_LEVEL_H