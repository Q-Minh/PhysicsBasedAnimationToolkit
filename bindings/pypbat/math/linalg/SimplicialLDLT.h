#ifndef PYPBAT_MATH_LINALG_SIMPLICIAL_LDLT_H
#define PYPBAT_MATH_LINALG_SIMPLICIAL_LDLT_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindSimplicialLDLT(pybind11::module& m);

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat

#endif // PYPBAT_MATH_LINALG_SIMPLICIAL_LDLT_H
