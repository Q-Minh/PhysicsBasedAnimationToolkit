#ifndef PYPBAT_GPU_GEOMETRY_SWEEP_AND_PRUNE_H
#define PYPBAT_GPU_GEOMETRY_SWEEP_AND_PRUNE_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindSweepAndPrune(nanobind::module_& m);

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_GEOMETRY_SWEEP_AND_PRUNE_H
