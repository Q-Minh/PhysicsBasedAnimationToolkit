#ifndef PYPBAT_FEM_LAPLACIAN_H
#define PYPBAT_FEM_LAPLACIAN_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace fem {

void BindLaplacian(nanobind::module_& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_LAPLACIAN_H
