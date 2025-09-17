#ifndef PYPBAT_GPU_COMMON_COMMON_H
#define PYPBAT_GPU_COMMON_COMMON_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace gpu {
namespace common {

void Bind(nanobind::module_& m);

} // namespace common
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_COMMON_COMMON_H
