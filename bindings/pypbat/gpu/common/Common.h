#ifndef PYPBAT_GPU_COMMON_COMMON_H
#define PYPBAT_GPU_COMMON_COMMON_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace gpu {
namespace common {

void Bind(pybind11::module& m);

} // namespace common
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_COMMON_COMMON_H
