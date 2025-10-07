#ifndef PYPBAT_GPU_PROFILING_PROFILING_H
#define PYPBAT_GPU_PROFILING_PROFILING_H

#include <nanobind/nanobind.h>

namespace pbat::py::gpu::profiling {

void Bind(nanobind::module_& m);

} // namespace pbat::py::gpu::profiling

#endif // PYPBAT_GPU_PROFILING_PROFILING_H
