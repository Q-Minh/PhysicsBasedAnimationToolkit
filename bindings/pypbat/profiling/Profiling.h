#ifndef PYPBAT_PROFILING_PROFILING_H
#define PYPBAT_PROFILING_PROFILING_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace profiling {

void Bind(nanobind::module_& m);

} // namespace profiling
} // namespace py
} // namespace pbat

#endif // PYPBAT_PROFILING_PROFILING_H
