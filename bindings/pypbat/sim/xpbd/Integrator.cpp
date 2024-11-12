#include "Integrator.h"

#include <pbat/sim/xpbd/Data.h>
#include <pbat/sim/xpbd/Enums.h>
#include <pbat/sim/xpbd/Integrator.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void BindIntegrator(pybind11::module& m)
{
    namespace pyb    = pybind11;
    using ScalarType = pbat::Scalar;
    using pbat::sim::xpbd::Data;
    using pbat::sim::xpbd::Integrator;
    pyb::class_<Integrator>(m, "Integrator")
        .def(
            pyb::init([](Data const& data,
                         [[maybe_unused]] std::size_t nMaxVertexTetrahedronOverlaps,
                         [[maybe_unused]] std::size_t nMaxVertexTriangleContacts) {
                return Integrator(data);
            }),
            pyb::arg("data"),
            pyb::arg("max_vertex_tetrahedron_overlaps") = 0,
            pyb::arg("max_vertex_triangle_contacts")    = 0,
            "Construct an XPBD integrator initialized with data. To access the data "
            "during simulation, go through the pbat.sim.xpbd.Integrator.data member.")
        .def(
            "step",
            &Integrator::Step,
            pyb::arg("dt"),
            pyb::arg("iterations"),
            pyb::arg("substeps") = 1,
            "Integrate the XPBD simulation 1 time step.")
        .def_property(
            "x",
            [](Integrator const& self) { return self.data.x; },
            [](Integrator& self, Eigen::Ref<MatrixX const> const& x) { self.data.x = x; },
            "3x|#nodes| nodal positions")
        .def_property(
            "v",
            [](Integrator const& self) { return self.data.v; },
            [](Integrator& self, Eigen::Ref<MatrixX const> const& v) { self.data.v = v; },
            "3x|#nodes| nodal velocities")
        .def_readwrite("data", &Integrator::data);
}

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat