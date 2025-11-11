/**
 * @file Device.h (Python bindings)
 */

#ifndef PYPBAT_GEOMETRY_DEVICE_H
#define PYPBAT_GEOMETRY_DEVICE_H

#include <nanobind/nanobind.h>

namespace pbat::py::geometry {

void BindDevice(nanobind::module_& m);

} // namespace pbat::py::geometry

#endif // PYPBAT_GEOMETRY_DEVICE_H
