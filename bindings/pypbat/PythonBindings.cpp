#include "pypbat/fem/fem.h"
#include "pypbat/profiling/Profiling.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(pypbat, m)
{
    m.doc()         = "Physics Based Animation Toolkit's python bindings";
    auto mprofiling = m.def_submodule("profiling");
    pypbat::bind_profiling(mprofiling);
    auto mfem = m.def_submodule("fem");
    pypbat::bind_fem(mfem);
}