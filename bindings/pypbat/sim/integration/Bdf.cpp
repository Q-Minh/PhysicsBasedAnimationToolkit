#include "Bdf.h"

#include <pbat/sim/integration/Bdf.h>
#include <pybind11/eigen.h>

namespace pbat::py::sim::integration {

void BindBdf(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::class_<pbat::sim::integration::Bdf>(m, "Bdf")
        .def(
            pyb::init([](int step, int order) { return pbat::sim::integration::Bdf(step, order); }),
            pyb::arg("step")  = 1,
            pyb::arg("order") = 2,
            "Construct a `step`-step BDF (backward differentiation formula) time integration "
            "scheme\n"
            "for a system of ODEs (ordinary differential equation) of order `order`.\n\n"
            "Args:\n"
            "    step (int): `0 < s < 7` backward differentiation scheme\n"
            "    order (int): `order > 0` order of the ODE\n"
            "Returns:\n"
            "    Bdf: BDF time integration scheme")
        .def_property(
            "order",
            &pbat::sim::integration::Bdf::Order,
            &pbat::sim::integration::Bdf::SetOrder,
            "Order of the ODE")
        .def_property(
            "s",
            [](pbat::sim::integration::Bdf& self) { return self.Step(); },
            &pbat::sim::integration::Bdf::SetStep,
            "Step `s` of the `s`-step BDF scheme")
        .def_property(
            "h",
            &pbat::sim::integration::Bdf::TimeStep,
            &pbat::sim::integration::Bdf::SetTimeStep,
            "Time step size")
        .def_readonly(
            "ti",
            &pbat::sim::integration::Bdf::ti,
            "Current time index s.t. `t = t0 + h ti`")
        .def(
            "state",
            [](pbat::sim::integration::Bdf& self, int k, int o) -> VectorX {
                return self.State(k, o);
            },
            pyb::arg("k"),
            pyb::arg("o") = 0,
            "o^th state derivative\n"
            "Args:\n"
            "    k (int): State index `k = 0, ..., s` for the vector `x^(o)(ti - s + k)`\n"
            "    o (int): Order of the state derivative `o = 0, ..., order - 1`\n"
            "Returns:\n"
            "    n x 1: state derivative vector `x^(o)(ti - s + k)`")
        .def(
            "current_state",
            [](pbat::sim::integration::Bdf& self, int o) -> VectorX {
                return self.CurrentState(o);
            },
            pyb::arg("o") = 0,
            "o^th most recent state derivative\n"
            "Args:\n"
            "    o (int): Order of the state derivative `o = 0, ..., order - 1`\n"
            "Returns:\n"
            "    n x 1: state derivative vector `x^(o)(ti - s + k)`")
        .def(
            "inertia",
            [](pbat::sim::integration::Bdf& self, int o) { return self.Inertia(o); },
            pyb::arg("o") = 0,
            "Inertia of the BDF scheme for the o^th state derivative\n"
            "Args:\n"
            "    o (int): Order of the state derivative `o = 0, ..., order - 1`\n"
            "Returns:\n"
            "    n x 1: inertia vector for the o^th state derivative")
        .def(
            "set_initial_conditions",
            [](pbat::sim::integration::Bdf& self, pyb::EigenDRef<MatrixX> x0) {
                self.SetInitialConditions(x0);
            },
            pyb::arg("x0"),
            "Set the initial conditions for the initial value problem\n"
            "Args:\n"
            "    x0 (n x order): matrix of initial conditions s.t. `x0.col(o) = x^(o)(t0)` for `o "
            "= 0, "
            "..., order - 1`\n"
            "Returns:\n"
            "    None")
        .def(
            "construct_equations",
            &pbat::sim::integration::Bdf::ConstructEquations,
            "Construct the BDF equations, i.e. compute `x^(o) = sum_{k=0}^{s-1} alpha_k x^(o)(ti - "
            "s + "
            "k)` for all `o = 0, ..., order - 1`\n"
            "Returns:\n"
            "    None")
        .def(
            "step",
            [](pbat::sim::integration::Bdf& self, pyb::EigenDRef<MatrixX> x) { self.Step(x); },
            pyb::arg("x"),
            "Advance the BDF scheme by one time step\n"
            "Args:\n"
            "    x (n x order): matrix of the current state derivatives s.t. `x.col(o) = "
            "x^(o)(ti)`\n"
            "Returns:\n"
            "    None")
        .def(
            "serialize",
            &pbat::sim::integration::Bdf::Serialize,
            pyb::arg("archive"),
            "Serialize the BDF scheme\n"
            "Args:\n"
            "    archive (Archive): Archive to serialize to\n"
            "Returns:\n"
            "    None")
        .def(
            "deserialize",
            &pbat::sim::integration::Bdf::Deserialize,
            pyb::arg("archive"),
            "Deserialize the BDF scheme\n"
            "Args:\n"
            "    archive (Archive): Archive to deserialize from\n"
            "Returns:\n"
            "    None");
}

} // namespace pbat::py::sim::integration