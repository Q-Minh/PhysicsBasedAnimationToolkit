#ifndef PYPBAT_GPU_COMMON_BUFFER_H
#define PYPBAT_GPU_COMMON_BUFFER_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace gpu {
namespace common {

void BindBuffer(nanobind::module_& m);

} // namespace common
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_COMMON_BUFFER_H
