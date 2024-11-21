#ifndef PBAT_SIM_VBD_SMOOTHER_H
#define PBAT_SIM_VBD_SMOOTHER_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Level;

struct Smoother
{
    Index iterations;

    void Apply(Level& L);
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_SMOOTHER_H