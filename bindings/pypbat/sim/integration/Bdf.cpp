#include "Bdf.h"

#include <nanobind/eigen/dense.h>
#include <pbat/sim/integration/Bdf.h>

namespace pbat::py::sim::integration {

void BindBdf(nanobind::module_& m)
{
    namespace nb  = nanobind;
    using BdfType = pbat::sim::integration::Bdf<Scalar>;
    nb::class_<BdfType>(m, "Bdf")
        .def(
            "__init__",
            [](BdfType* self, int step, int order) { new (self) BdfType(step, order); },
            nb::arg("step")  = 1,
            nb::arg("order") = 2,
            "Construct a `step`-step BDF (backward differentiation formula) time integration "
            "scheme\n"
            "for a system of ODEs (ordinary differential equation) of order `order`.\n\n"
            "Args:\n"
            "    step (int): `0 < s < 7` backward differentiation scheme\n"
            "    order (int): `order > 0` order of the ODE\n"
            "Returns:\n"
            "    Bdf: BDF time integration scheme")
        .def_prop_rw("order", &BdfType::Order, &BdfType::SetOrder, "Order of the ODE")
        .def_prop_rw(
            "s",
            [](BdfType& self) { return self.Step(); },
            &BdfType::SetStep,
            "Step `s` of the `s`-step BDF scheme")
        .def_prop_rw("h", &BdfType::TimeStep, &BdfType::SetTimeStep, "Time step size")
        .def_ro("ti", &BdfType::ti, "Current time index s.t. `t = t0 + h ti`")
        .def(
            "state",
            [](BdfType& self, int k, int o) -> VectorX { return self.State(k, o); },
            nb::arg("k"),
            nb::arg("o") = 0,
            "o^th state derivative\n"
            "Args:\n"
            "    k (int): State index `k = 0, ..., s` for the vector `x^(o)(ti - s + k)`\n"
            "    o (int): Order of the state derivative `o = 0, ..., order - 1`\n"
            "Returns:\n"
            "    n x 1: state derivative vector `x^(o)(ti - s + k)`")
        .def(
            "current_state",
            [](BdfType& self, int o) -> VectorX { return self.CurrentState(o); },
            nb::arg("o") = 0,
            "o^th most recent state derivative\n"
            "Args:\n"
            "    o (int): Order of the state derivative `o = 0, ..., order - 1`\n"
            "Returns:\n"
            "    n x 1: state derivative vector `x^(o)(ti - s + k)`")
        .def(
            "inertia",
            [](BdfType& self, int o) { return self.Inertia(o); },
            nb::arg("o") = 0,
            "Inertia of the BDF scheme for the o^th state derivative\n"
            "Args:\n"
            "    o (int): Order of the state derivative `o = 0, ..., order - 1`\n"
            "Returns:\n"
            "    n x 1: inertia vector for the o^th state derivative")
        .def(
            "set_initial_conditions",
            [](BdfType& self, nb::DRef<MatrixX> x0) { self.SetInitialConditions(x0); },
            nb::arg("x0"),
            "Set the initial conditions for the initial value problem\n"
            "Args:\n"
            "    x0 (n x order): matrix of initial conditions s.t. `x0.col(o) = x^(o)(t0)` for `o "
            "= 0, "
            "..., order - 1`\n"
            "Returns:\n"
            "    None")
        .def(
            "construct_equations",
            &BdfType::ConstructEquations,
            "Construct the BDF equations, i.e. compute `x^(o) = sum_{k=0}^{s-1} alpha_k x^(o)(ti - "
            "s + k)` for all `o = 0, ..., order - 1`\n"
            "Returns:\n"
            "    None")
        .def(
            "step",
            [](BdfType& self, nb::DRef<MatrixX> x) { self.Step(x); },
            nb::arg("x"),
            "Advance the BDF scheme by one time step\n"
            "Args:\n"
            "    x (n x order): matrix of the current state derivatives s.t. `x.col(o) = "
            "x^(o)(ti)`\n"
            "Returns:\n"
            "    None")
        .def(
            "serialize",
            &BdfType::Serialize,
            nb::arg("archive"),
            "Serialize the BDF scheme\n"
            "Args:\n"
            "    archive (Archive): Archive to serialize to\n"
            "Returns:\n"
            "    None")
        .def(
            "deserialize",
            &BdfType::Deserialize,
            nb::arg("archive"),
            "Deserialize the BDF scheme\n"
            "Args:\n"
            "    archive (Archive): Archive to deserialize from\n"
            "Returns:\n"
            "    None");
}

} // namespace pbat::py::sim::integration