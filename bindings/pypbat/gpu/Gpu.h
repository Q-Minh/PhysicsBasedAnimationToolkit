#ifndef PYPBAT_GPU_GPU_H
#define PYPBAT_GPU_GPU_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace gpu {

void Bind(pybind11::module& m);

} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_GPU_H
