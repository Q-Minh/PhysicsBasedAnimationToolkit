#include "Gpu.h"

#include "geometry/Geometry.h"
#include "xpbd/Xpbd.h"

namespace pbat {
namespace py {
namespace gpu {

void Bind(pybind11::module& m)
{
    namespace pyb  = pybind11;
    auto mgeometry = m.def_submodule("geometry");
    geometry::Bind(mgeometry);
    auto mxpbd = m.def_submodule("xpbd");
    xpbd::Bind(mxpbd);
}

} // namespace gpu
} // namespace py
} // namespace pbat