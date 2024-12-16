#ifndef PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H
#define PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H

#include "Level.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"

#include <vector>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Hierarchy
{
    Hierarchy(
        Data data,
        std::vector<VolumeMesh> cages,
        IndexVectorX const& cycle  = {},
        IndexVectorX const& siters = {});

    Data data;                 ///< Root level
    std::vector<Level> levels; ///< Coarse levels
    IndexVectorX cycle; ///< |#level visits| ordered array of levels to visit during the solve.
                        ///< Level -1 is the root, 0 the first coarse level, etc.
    IndexVectorX
        siters; ///< |#cages+1| max smoother iterations at each level, starting from the root
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H