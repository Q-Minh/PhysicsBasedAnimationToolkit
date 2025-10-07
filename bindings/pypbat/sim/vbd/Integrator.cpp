#include "Integrator.h"

#include <memory>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/unique_ptr.h>
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
    namespace nb = nanobind;
    using pbat::sim::vbd::AndersonIntegrator;
    using pbat::sim::vbd::BroydenIntegrator;
    using pbat::sim::vbd::ChebyshevIntegrator;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EAccelerationStrategy;
    using pbat::sim::vbd::EInitializationStrategy;
    using pbat::sim::vbd::Integrator;
    using pbat::sim::vbd::NesterovIntegrator;
    nb::class_<Integrator>(m, "Integrator")
        .def(nb::new_([](Data const& data) -> std::unique_ptr<Integrator, nb::deleter<Integrator>> {
            if (data.eAcceleration == EAccelerationStrategy::Chebyshev)
                return std::unique_ptr<ChebyshevIntegrator, nb::deleter<Integrator>>(
                    new ChebyshevIntegrator(data));
            if (data.eAcceleration == EAccelerationStrategy::Anderson)
                return std::unique_ptr<AndersonIntegrator, nb::deleter<Integrator>>(
                    new AndersonIntegrator(data));
            if (data.eAcceleration == EAccelerationStrategy::Broyden)
                return std::unique_ptr<BroydenIntegrator, nb::deleter<Integrator>>(
                    new BroydenIntegrator(data));
            if (data.eAcceleration == EAccelerationStrategy::Nesterov)
                return std::unique_ptr<NesterovIntegrator, nb::deleter<Integrator>>(
                    new NesterovIntegrator(data));
            return std::unique_ptr<Integrator, nb::deleter<Integrator>>(new Integrator(data));
        }))
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