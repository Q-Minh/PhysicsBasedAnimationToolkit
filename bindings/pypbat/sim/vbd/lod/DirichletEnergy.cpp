#include "DirichletEnergy.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/lod/DirichletEnergy.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindDirichletEnergy(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::lod::DirichletEnergy;
    using pbat::sim::vbd::lod::DirichletQuadrature;
    pyb::class_<DirichletEnergy>(m, "DirichletEnergy")
        .def(
            pyb::init([](Data const& problem, DirichletQuadrature const& DQ) {
                return DirichletEnergy(problem, DQ);
            }),
            pyb::arg("problem"),
            pyb::arg("dirichlet_quadrature"))
        .def_readwrite("muD", &DirichletEnergy::muD)
        .def_readwrite("dg", &DirichletEnergy::dg);
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat