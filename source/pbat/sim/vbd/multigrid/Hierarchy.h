#ifndef PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H
#define PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H

#include "Level.h"
#include "Prolongation.h"
#include "Quadrature.h"
#include "Restriction.h"
#include "pbat/common/Hash.h"
#include "pbat/sim/vbd/Data.h"

#include <unordered_map>
#include <variant>
#include <vector>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Hierarchy
{
    /**
     * @brief
     * @param root Problem defined on the finest mesh
     * @param X Ordered list of coarse embedding/cage mesh vertex positions
     * @param E Ordered list of coarse embedding/cage mesh tetrahedral elements
     * @param cycle 2x|#transitions| ordered list of transitions, such that the pair (cycle(0,t),
     * cycle(1,t)) represents the transition from level cycle(0,t) to level cycle(1,t). The root
     * level has index -1 and coarse levels start from index 0.
     * @param transitionSchedule |#transitions| list of iterations to spend on each transition.
     * @param smoothingSchedule <|#transitions+1| list of iterations to spend on each visited level
     * in the cycle.
     */
    Hierarchy(
        Data root,
        std::vector<Eigen::Ref<MatrixX const>> const& X,
        std::vector<Eigen::Ref<IndexMatrixX const>> const& E,
        Eigen::Ref<IndexMatrixX const> const& cycle                 = IndexMatrixX{},
        Eigen::Ref<IndexVectorX const> const& transitionSchedule    = IndexVectorX{},
        Eigen::Ref<IndexVectorX const> const& smoothingSchedule     = IndexVectorX{},
        std::vector<CageQuadratureParameters> const& cageQuadParams = {});

    Data mRoot;                 ///< Finest mesh
    std::vector<Level> mLevels; ///< Ordered list of coarse embedding/cage meshes
    IndexMatrixX mCycle; ///< 2x|#transitions| ordered list of transitions, such that the pair
                         ///< (cycle(0,t), cycle(1,t)) represents the transition from level
    ///< cycle(0,t) to level cycle(1,t). The root level has index -1 and coarse
    ///< levels start from index 0.
    IndexVectorX
        mTransitionSchedule; ///< |#transitions| list of iterations to spend on each transition.
    IndexVectorX mSmoothingSchedule; ///< <|#transitions+1| list of iterations to spend on each
                                     ///< visited level in the cycle.

    using Transition = std::variant<Restriction, Prolongation>;
    std::unordered_map<IndexVector<2>, Transition> mTransitions; ///< Transition operators
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H