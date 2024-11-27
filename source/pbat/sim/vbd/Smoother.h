#ifndef PBAT_SIM_VBD_SMOOTHER_H
#define PBAT_SIM_VBD_SMOOTHER_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Data;
struct Level;

struct Smoother
{
    Index iterations{10}; ///< Number of smoothing iterations

    /**
     * @brief Smooth level L
     *
     * @param dt
     * @param L
     */
    void Apply(Scalar dt, Level& L);
    /**
     * @brief Smooth root level
     *
     * @param dt
     * @param data
     */
    void Apply(Scalar dt, Data& data);
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_SMOOTHER_H