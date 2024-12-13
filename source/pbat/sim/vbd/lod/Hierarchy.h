#ifndef PBAT_SIM_VBD_LOD_HIERARCHY_H
#define PBAT_SIM_VBD_LOD_HIERARCHY_H

#include "Level.h"
#include "Prolongation.h"
#include "Quadrature.h"
#include "Restriction.h"
#include "pbat/common/Hash.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"

#include <unordered_map>
#include <variant>
#include <vector>

namespace pbat {
namespace sim {
namespace vbd {
namespace lod {

struct Hierarchy
{
    /**
     * @brief
     * @param root Problem defined on the finest mesh
     * @param cages Ordered list of coarse embedding/cage meshes
     * @param cycle |#transitions+1| ordered list of transitions, such that the pair (cycle(t),
     * cycle(t+1)) represents the transition from level cycle(t) to level cycle(t+1). The root level
     * has index -1 and coarse levels start from index 0.
     * @param transitionSchedule |#transitions| list of iterations to spend on each transition.
     * @param smoothingSchedule <|#transitions+1| list of iterations to spend on each visited level
     * in the cycle.
     */
    Hierarchy(
        Data root,
        std::vector<VolumeMesh> const& cages,
        Eigen::Ref<IndexVectorX const> const& cycle                 = IndexVectorX{},
        Eigen::Ref<IndexVectorX const> const& smoothingSchedule     = IndexVectorX{},
        Eigen::Ref<IndexVectorX const> const& transitionSchedule    = IndexVectorX{},
        std::vector<CageQuadratureParameters> const& cageQuadParams = {});

    Data mRoot;                 ///< Finest mesh
    std::vector<Level> mLevels; ///< Ordered list of coarse embedding/cage meshes
    IndexVectorX
        mCycle; ///< |#transitions+1| ordered list of transitions, such that the pair
                ///< (cycle(t), cycle(t+1)) represents the transition from level cycle(t) to level
                ///< cycle(t+1). The root level has index -1 and coarse levels start from index 0.
    IndexVectorX mSmoothingSchedule; ///< <|#transitions+1| list of iterations to spend on each
                                     ///< visited level in the cycle.
    IndexVectorX
        mTransitionSchedule; ///< |#transitions| list of iterations to spend on each transition.

    using Transition = std::variant<Restriction, Prolongation>;
    std::unordered_map<IndexVector<2>, Transition> mTransitions; ///< Transition operators
};

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LOD_HIERARCHY_H