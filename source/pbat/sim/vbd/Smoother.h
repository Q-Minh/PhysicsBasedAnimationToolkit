#ifndef PBAT_SIM_VBD_SMOOTHER_H
#define PBAT_SIM_VBD_SMOOTHER_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Level;

struct Smoother
{
    Index iterations{10}; ///< Number of smoothing iterations

    /**
     * @brief Smooth level L
     *
     * @param L
     */
    void Apply(Level& L);
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_SMOOTHER_H