#ifndef PYPBAT_GPU_COMMON_BUFFER_H
#define PYPBAT_GPU_COMMON_BUFFER_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace gpu {
namespace common {

void BindBuffer(pybind11::module& m);

} // namespace common
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_COMMON_BUFFER_H
