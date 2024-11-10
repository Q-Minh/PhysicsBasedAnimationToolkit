#include "Integrator.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Enums.h>
#include <pbat/sim/vbd/Integrator.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindIntegrator(pybind11::module& m)
{
    namespace pyb    = pybind11;
    using ScalarType = pbat::Scalar;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EInitializationStrategy;
    using pbat::sim::vbd::Integrator;
    pyb::class_<Integrator>(m, "Integrator")
        .def(
            pyb::init([](Data& data) { return Integrator(std::move(data)); }),
            "Construct a VBD integrator initialized with data. The passed in data is 'moved' in "
            "the C++ sense, i.e. the C++ side will take ownership of the data. To access the data "
            "during simulation, go through the pbat.sim.vbd.Integrator.data member.")
        .def(
            "step",
            &Integrator::Step,
            pyb::arg("dt"),
            pyb::arg("iterations"),
            pyb::arg("substeps") = 1,
            pyb::arg("rho")      = ScalarType(1),
            "Integrate the VBD simulation 1 time step.")
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
        .def_property(
            "strategy",
            [](Integrator const& self) { return self.data.strategy; },
            [](Integrator& self, EInitializationStrategy strategy) {
                self.data.strategy = strategy;
            },
            "Initialization strategy for non-linear optimization solve.")
        .def_property(
            "kD",
            [](Integrator const& self) { return self.data.kD; },
            [](Integrator& self, Scalar kD) { self.data.kD = kD; },
            "Rayleigh damping coefficient.")
        .def_property(
            "detH_residual",
            [](Integrator const& self) { return self.data.detHZero; },
            [](Integrator& self, Scalar detHZero) { self.data.detHZero = detHZero; },
            "Numerical zero used in 'singular' hessian determinant check.")
        .def_readwrite("data", &Integrator::data);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat