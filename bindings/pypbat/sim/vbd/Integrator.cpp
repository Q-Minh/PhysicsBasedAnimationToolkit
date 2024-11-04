#include "Integrator.h"

#include <pbat/sim/vbd/Data.h>
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
        .def_readwrite("data", &Integrator::data);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat