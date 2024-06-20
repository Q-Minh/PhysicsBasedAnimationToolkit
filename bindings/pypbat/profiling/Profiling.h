#ifndef PYPBAT_PROFILING_PROFILING_H
#define PYPBAT_PROFILING_PROFILING_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace profiling {

void Bind(pybind11::module& m);

} // namespace profiling
} // namespace py
} // namespace pbat

#endif // PYPBAT_PROFILING_PROFILING_H
