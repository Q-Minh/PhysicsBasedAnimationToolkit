#ifndef PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H
#define PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H

#include "pbat/Aliases.h"

#include <vector>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Hierarchy;

struct HyperReduction
{
    HyperReduction(Hierarchy const& hierarchy, Index clusterSize = 5);

    /**
     * @brief Hierarchical clustering of mesh elements
     */
    std::vector<IndexVectorX> Cptr;
    std::vector<IndexVectorX> Cadj;

    std::vector<VectorX>
        Ep;       ///< |#levels+1| linear polynomial errors at fine elements and at each level
    Scalar EpMax; ///< Maximum allowable linear polynomial error in any cluster
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H
