#ifndef PYPBAT_GPU_XPBD_XPBD_H
#define PYPBAT_GPU_XPBD_XPBD_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace gpu {
namespace xpbd {

void Bind(pybind11::module& m);

} // namespace xpbd
} // namespace gpu
} // namespace py
} // namespace pbat

#endif // PYPBAT_GPU_XPBD_XPBD_H
