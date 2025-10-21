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
            [](BdfType& self) { return self.GetStep(); },
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
            "    None")
        .doc() =
        R"doc(
BDF (Backward Differentiation Formula) time integration scheme for a system of ODEs.

The Bdf class implements a backward differentiation formula time integration scheme 
for solving initial value problems (IVPs) involving systems of ordinary differential 
equations (ODEs). 

Background
----------
An order `p` system of ODEs x^(p) = f(t, x^(p-1), ..., x) can be transformed into 
a system of `p` first-order ODEs using slack variables x^(o) = d^o/dt^o x for 
o = 0, ..., p-1.

The BDF scheme discretizes each equation as:
    sum_{k=0}^s alpha_k x_{n-s+k} = h * beta * f(t, x_n^(p-1), ..., x_n)

This results in a system of equations that can be solved for the states and their 
derivatives at each time step using root-finding or numerical optimization.

Key Features
------------
- Supports s-step BDF schemes where 0 < s < 7
- Handles ODEs of arbitrary order
- Stores past states and their derivatives automatically
- Provides inertia terms for equation construction
- Includes serialization/deserialization capabilities

Usage Pattern
-------------
The typical usage pattern for time integration is:

    # Initialize BDF scheme
    bdf = Bdf(step=2, order=2)  # 2-step BDF for 2nd order ODE
    bdf.h = 0.01  # Set time step size
    # Set initial conditions
    bdf.set_initial_conditions(x0)  # x0 is n x order matrix
    # Time stepping loop
    for _ in range(num_steps):
        bdf.construct_equations()  # Compute inertia terms
        # Solve user-defined equations here using:
        # - bdf.inertia(o) for o-th derivative inertia
        # - bdf.current_state(o) for current o-th derivative
        x_new = solve_user_equations(bdf)  # User implementation
        bdf.step(x_new)  # Advance to next time step

Parameters
----------
step : int, default=1
    Step `s` of the s-step BDF scheme (0 < s < 7). Higher values provide 
    better accuracy but require more storage and may be less stable.
    
order : int, default=2  
    Order of the ODE system (order > 0). For example, order=2 for second-order
    systems like Newton's equations of motion.

Attributes
----------
h : float
    Time step size. Must be positive.
    
ti : int (read-only)
    Current time index such that t = t0 + h * ti.
    
order : int
    Order of the ODE system. Can be modified after construction.
    
s : int
    Step of the BDF scheme. Can be modified after construction.

Examples
--------
    import numpy as np
    # Set up 2-step BDF for 2nd order system
    bdf = Bdf(step=2, order=2)
    bdf.h = 0.01
    # Initial conditions: x0 = position, v0 = velocity  
    x0 = np.array([[1.0], [0.0]])  # position, velocity
    bdf.set_initial_conditions(x0)
    # Time integration
    for _ in range(1000):
        bdf.construct_equations()
        # Get current state and inertia
        x_inertia = bdf.inertia(0)  # position inertia
        v_inertia = bdf.inertia(1)  # velocity inertia
        # Solve: M*a = F - K*x - C*v (user-specific)
        # where BDF provides the inertia terms
        x_new, v_new = solve_dynamics(bdf, forces)
        # Advance time step
        x_solution = np.column_stack([x_new, v_new])
        bdf.step(x_solution)
)doc";
}

} // namespace pbat::py::sim::integration