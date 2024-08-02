#include "Gpu.h"

#include "geometry/Geometry.h"

namespace pbat {
namespace py {
namespace gpu {

void Bind(pybind11::module& m)
{
    namespace pyb  = pybind11;
    auto mgeometry = m.def_submodule("geometry");
    geometry::Bind(mgeometry);
}

} // namespace gpu
} // namespace py
} // namespace pbat