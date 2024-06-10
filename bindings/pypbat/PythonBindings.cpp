#include "fem/fem.h"
#include "profiling/Profiling.h"

#include <pybind11/pybind11.h>

PYBIND11_MODULE(_pbat, m)
{
    namespace pyb = pybind11;
    m.doc()         = "Physics Based Animation Toolkit's python bindings";
    auto mprofiling = m.def_submodule("profiling");
    pbat::py::profiling::Bind(mprofiling);
    auto mfem = m.def_submodule("fem");
    pbat::py::fem::Bind(mfem);
}