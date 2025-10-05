#include "fem/Fem.h"
#include "geometry/Geometry.h"
#include "gpu/Gpu.h"
#include "graph/Graph.h"
#include "io/Io.h"
#include "math/Math.h"
#include "profiling/Profiling.h"
#include "sim/Sim.h"

#include <nanobind/nanobind.h>

NB_MODULE(_pbat, m)
{
    namespace nb    = nanobind;
    m.doc()         = "Physics Based Animation Toolkit's python bindings";
    auto mprofiling = m.def_submodule("profiling");
    pbat::py::profiling::Bind(mprofiling);
    auto mgeometry = m.def_submodule("geometry");
    pbat::py::geometry::Bind(mgeometry);
    auto mgraph = m.def_submodule("graph");
    pbat::py::graph::Bind(mgraph);
    auto mio = m.def_submodule("io");
    pbat::py::io::Bind(mio);
    auto mmath = m.def_submodule("math");
    pbat::py::math::Bind(mmath);
    auto mfem = m.def_submodule("fem");
    pbat::py::fem::Bind(mfem);
    auto msim = m.def_submodule("sim");
    pbat::py::sim::Bind(msim);
    // Bind GPU module at the end, since it is layered on top of non-GPU modules
    auto mgpu = m.def_submodule("gpu");
    pbat::py::gpu::Bind(mgpu);
}