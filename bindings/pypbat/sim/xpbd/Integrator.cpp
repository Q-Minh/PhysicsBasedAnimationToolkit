#include "Integrator.h"

#include <nanobind/eigen/dense.h>
#include <pbat/sim/xpbd/Data.h>
#include <pbat/sim/xpbd/Enums.h>
#include <pbat/sim/xpbd/Integrator.h>

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void BindIntegrator(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = pbat::Scalar;
    using pbat::sim::xpbd::Data;
    using pbat::sim::xpbd::Integrator;
    nb::class_<Integrator>(m, "Integrator")
        .def(
            "__init__",
            [](Integrator* self,
               Data const& data,
               [[maybe_unused]] std::size_t nMaxVertexTetrahedronOverlaps,
               [[maybe_unused]] std::size_t nMaxVertexTriangleContacts) {
                new (self) Integrator(data);
            },
            nb::arg("data"),
            nb::arg("max_vertex_tetrahedron_overlaps") = 0,
            nb::arg("max_vertex_triangle_contacts")    = 0,
            "Construct an XPBD integrator initialized with data. To access the data "
            "during simulation, go through the pbat.sim.xpbd.Integrator.data member.")
        .def(
            "step",
            &Integrator::Step,
            nb::arg("dt"),
            nb::arg("iterations"),
            nb::arg("substeps") = 1,
            "Integrate the XPBD simulation 1 time step.")
        .def_prop_rw(
            "x",
            [](Integrator const& self) { return self.data.x; },
            [](Integrator& self, Eigen::Ref<MatrixX const> const& x) { self.data.x = x; },
            "3x|#nodes| nodal positions")
        .def_prop_rw(
            "v",
            [](Integrator const& self) { return self.data.v; },
            [](Integrator& self, Eigen::Ref<MatrixX const> const& v) { self.data.v = v; },
            "3x|#nodes| nodal velocities")
        .def_rw("data", &Integrator::data);
}

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat