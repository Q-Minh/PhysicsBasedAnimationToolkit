#include <pbat/fem/Mesh.h>
#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void bind_mesh(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat