#include "Integrator.h"

#include <memory>
#include <nanobind/eigen/dense.h>
#include <pbat/sim/vbd/AndersonIntegrator.h>
#include <pbat/sim/vbd/BroydenIntegrator.h>
#include <pbat/sim/vbd/ChebyshevIntegrator.h>
#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Enums.h>
#include <pbat/sim/vbd/Integrator.h>
#include <pbat/sim/vbd/NesterovIntegrator.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindIntegrator(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = pbat::Scalar;
    using pbat::sim::vbd::AndersonIntegrator;
    using pbat::sim::vbd::BroydenIntegrator;
    using pbat::sim::vbd::ChebyshevIntegrator;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EAccelerationStrategy;
    using pbat::sim::vbd::EInitializationStrategy;
    using pbat::sim::vbd::Integrator;
    using pbat::sim::vbd::NesterovIntegrator;
    nb::class_<Integrator>(m, "Integrator")
        .def(
            "__init__",
            [](Integrator* self, Data const& data) {
                // ERROR:
                // Placement new does not work with polymorphism here, it will always 
                // only construct the base Integrator.
                if (data.eAcceleration == EAccelerationStrategy::Chebyshev)
                    new (self) ChebyshevIntegrator(data);
                if (data.eAcceleration == EAccelerationStrategy::Anderson)
                    new (self) AndersonIntegrator(data);
                if (data.eAcceleration == EAccelerationStrategy::Broyden)
                    new (self) BroydenIntegrator(data);
                if (data.eAcceleration == EAccelerationStrategy::Nesterov)
                    new (self) NesterovIntegrator(data);
                new (self) Integrator(data);
            },
            "Construct a VBD integrator initialized with data. To access the data "
            "during simulation, go through the pbat.sim.vbd.Integrator.data member.")
        .def(
            "step",
            &Integrator::Step,
            nb::arg("dt"),
            nb::arg("iterations"),
            nb::arg("substeps") = 1,
            "Integrate the VBD simulation 1 time step.\n\n"
            "Args:\n"
            "    dt (float): Time step size.\n"
            "    iterations (int): Number of iterations to solve the non-linear optimization "
            "problem.\n"
            "    substeps (int): Number of substeps to take per time step.")
        .def(
            "trace_next_step",
            &Integrator::TraceNextStep,
            nb::arg("path") = ".",
            nb::arg("t")    = -1)
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
        .def_prop_rw(
            "strategy",
            [](Integrator const& self) { return self.data.strategy; },
            [](Integrator& self, EInitializationStrategy strategy) {
                self.data.strategy = strategy;
            },
            "Acceleration strategy")
        .def_prop_rw(
            "kD",
            [](Integrator const& self) { return self.data.kD; },
            [](Integrator& self, Scalar kD) { self.data.kD = kD; },
            "Rayleigh damping coefficient")
        .def_prop_rw(
            "detH_residual",
            [](Integrator const& self) { return self.data.detHZero; },
            [](Integrator& self, Scalar detHZero) { self.data.detHZero = detHZero; },
            "Determinant of the residual Hessian for pseudo singularity check")
        .def_rw("data", &Integrator::data);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat