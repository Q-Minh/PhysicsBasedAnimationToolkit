#include "pypbat/fem/fem.h"
#include "pypbat/profiling/Profiling.h"

#include <pybind11/pybind11.h>

namespace pyb = pybind11;

PYBIND11_MODULE(pypbat, m)
{
    m.doc()         = "Physics Based Animation Toolkit's python bindings";
    auto mprofiling = m.def_submodule("profiling");
    pbat::py::profiling::Bind(mprofiling);
    auto mfem = m.def_submodule("fem");
    pbat::py::fem::Bind(mfem);
}