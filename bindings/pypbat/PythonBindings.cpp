#include "fem/Fem.h"
#include "geometry/Geometry.h"
#include "gpu/Gpu.h"
#include "math/Math.h"
#include "profiling/Profiling.h"
#include "sim/Sim.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(_pbat, m)
{
    namespace pyb   = pybind11;
    m.doc()         = "Physics Based Animation Toolkit's python bindings";
    auto mprofiling = m.def_submodule("profiling");
    pbat::py::profiling::Bind(mprofiling);
    auto mfem = m.def_submodule("fem");
    pbat::py::fem::Bind(mfem);
    auto mgeometry = m.def_submodule("geometry");
    pbat::py::geometry::Bind(mgeometry);
    auto mmath = m.def_submodule("math");
    pbat::py::math::Bind(mmath);
    auto msim = m.def_submodule("sim");
    pbat::py::sim::Bind(msim);
    // Bind GPU module at the end, since it is layered on top of non-GPU modules
    auto mgpu = m.def_submodule("gpu");
    pbat::py::gpu::Bind(mgpu);
}