#include "DirichletEnergy.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/multigrid/DirichletEnergy.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindDirichletEnergy(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::multigrid::DirichletEnergy;
    using pbat::sim::vbd::multigrid::DirichletQuadrature;
    using pbat::sim::vbd::multigrid::VolumeMesh;
    pyb::class_<DirichletEnergy>(m, "DirichletEnergy")
        .def(pyb::init(
            [](Data const& problem, pbat::py::fem::Mesh const& CM, DirichletQuadrature const& DQ) {
                return DirichletEnergy(problem, *CM.Raw<VolumeMesh>(), DQ);
            }),
            pyb::arg("problem"),
            pyb::arg("cage_mesh"),
            pyb::arg("dirichlet_quadrature"))
        .def_readwrite("muD", &DirichletEnergy::muD)
        .def_readwrite("Ncg", &DirichletEnergy::Ncg)
        .def_readwrite("dg", &DirichletEnergy::dg);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat