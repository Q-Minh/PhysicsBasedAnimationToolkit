#ifndef PBAT_SIM_VBD_LOD_SMOOTHER_H
#define PBAT_SIM_VBD_LOD_SMOOTHER_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Data;

namespace lod {

struct Level;

struct Smoother
{
    void Apply(Index iters, Scalar dt, Level& l) const;
    void Apply(Index iters, Scalar dt, Data& root) const;
};

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_LOD_SMOOTHER_H