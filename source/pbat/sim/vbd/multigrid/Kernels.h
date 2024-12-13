#ifndef PBAT_SIM_VBD_MULTIGRID_KERNELS_H
#define PBAT_SIM_VBD_MULTIGRID_KERNELS_H

#include "pbat/Aliases.h"
#include "pbat/HostDevice.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/HyperElasticity.h"
#include "pbat/sim/vbd/Kernels.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {
namespace kernels {

namespace mini = math::linalg::mini;

} // namespace kernels
} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_KERNELS_H