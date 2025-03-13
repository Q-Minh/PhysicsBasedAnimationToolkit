#include "Integrator.h"

#include <memory>
#include <pbat/sim/vbd/ChebyshevIntegrator.h>
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
    using pbat::sim::vbd::ChebyshevIntegrator;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EAccelerationStrategy;
    using pbat::sim::vbd::EInitializationStrategy;
    using pbat::sim::vbd::Integrator;
    pyb::class_<Integrator>(m, "Integrator")
        .def(
            pyb::init([](Data const& data) -> std::unique_ptr<Integrator> {
                if (data.eAcceleration == EAccelerationStrategy::Chebyshev)
                    return std::make_unique<ChebyshevIntegrator>(data);
                return std::make_unique<Integrator>(data);
            }),
            "Construct a VBD integrator initialized with data. To access the data "
            "during simulation, go through the pbat.sim.vbd.Integrator.data member.")
        .def(
            "step",
            &Integrator::Step,
            pyb::arg("dt"),
            pyb::arg("iterations"),
            pyb::arg("substeps") = 1,
            "Integrate the VBD simulation 1 time step.\n\n"
            "Args:\n"
            "    dt (float): Time step size.\n"
            "    iterations (int): Number of iterations to solve the non-linear optimization "
            "problem.\n"
            "    substeps (int): Number of substeps to take per time step.")
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

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat