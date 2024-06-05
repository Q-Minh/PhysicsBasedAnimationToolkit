#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pypbat {

void bind_fem(py::module& m);

} // namespace pypbat