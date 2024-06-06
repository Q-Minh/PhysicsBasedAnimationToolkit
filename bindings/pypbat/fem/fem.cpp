#include "fem.h"

#include "Mesh.h"

namespace pbat {
namespace py {
namespace fem {

namespace pyb = pybind11;

void bind(pyb::module& m)
{
    m.doc() = "Finite Element Method module";
    bind_mesh(m);
}

} // namespace fem
} // namespace py
} // namespace pbat