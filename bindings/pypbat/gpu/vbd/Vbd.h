#ifndef PYPBAT_GPU_VBD_H
#define PYPBAT_GPU_VBD_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace gpu {
namespace vbd {

void Bind(pybind11::module& m);

} // namespace vbd
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_VBD_H
