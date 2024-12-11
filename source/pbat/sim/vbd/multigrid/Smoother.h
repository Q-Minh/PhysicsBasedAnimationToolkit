#ifndef PBAT_SIM_VBD_MULTIGRID_SMOOTHER_H
#define PBAT_SIM_VBD_MULTIGRID_SMOOTHER_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Data;

namespace multigrid {

struct Level;

struct Smoother
{
    void Apply(Index iters, Scalar dt, Level& l);
    void Apply(Index iters, Scalar dt, Data& root);
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_SMOOTHER_H