#ifndef PBAT_SIM_VBD_HIERARCHY_H
#define PBAT_SIM_VBD_HIERARCHY_H

#include "Data.h"
#include "Level.h"
#include "Prolongation.h"
#include "Restriction.h"
#include "Smoother.h"

#include <memory>
#include <variant>
#include <vector>

namespace pbat {
namespace sim {
namespace vbd {

struct Hierarchy
{
    using Transition =
        std::variant<Restriction, Prolongation>; ///< A Transition between levels must be either a
                                                 ///< Restriction or a Prolongation operator.

    /**
     * @brief Construct a new Hierarchy object
     *
     * @param root Root problem
     * @param levels Ordered coarsened levels, such that li > lj => li is coarser than lj for levels
     * li, lj
     * @param smoothers Smoothers associated with each level
     * @param Nl Root level shape functions evaluated at quadrature poitns of each coarse level
     * @param transitions The multilevel solver cycle scheme as an ordered list of transitions
     * (either restrictions or prolongations)
     */
    Hierarchy(
        Data root,
        std::vector<Level> levels,
        std::vector<Smoother> smoothers,
        std::vector<MatrixX> Nl,
        std::vector<Transition> transitions);

    Data root;                 ///< Root level, i.e. the finest resolution
    std::vector<Level> levels; ///< List of the hierarchy's levels, such that levels[l] is finer
                               ///< than levels[l+k], k > 0.
    std::vector<Smoother> smoothers; ///< |#levels| list of smoothers associated with each level.
    std::vector<MatrixX>
        Nl; ///< List of 4x|#quad.pts at level l| arrays of root level shape functions evaluated at
            ///< quadrature points of the corresponding level. This allows transferring problem
            ///< parameters from the root directly to a given coarser level, for example inertial
            ///< targets xtildeg.
    std::vector<Transition>
        transitions; ///< Ordered list of level transitions, such that traversing transitions from 0
                     ///< to transitions.size()-1 executes the complete multigrid solver cycle.
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_HIERARCHY_H