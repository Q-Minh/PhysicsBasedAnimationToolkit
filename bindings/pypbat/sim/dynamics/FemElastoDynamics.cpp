#include "FemElastoDynamics.h"

#include <nanobind/eigen/dense.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/Tetrahedron.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pbat/sim/dynamics/FemElastoDynamics.h>

namespace pbat::py::sim::dynamics {

void BindFemElastoDynamics([[maybe_unused]] nanobind::module_& m)
{
    namespace nb            = nanobind;
    using ScalarType        = Scalar;
    using IndexType         = Index;
    constexpr int kDims     = 3;
    using ElementType       = fem::Tetrahedron<1>;
    using ElasticEnergyType = physics::StableNeoHookeanEnergy<kDims>;
    using ElastoDynamics    = pbat::sim::dynamics::
        FemElastoDynamics<ElementType, kDims, ElasticEnergyType, ScalarType, IndexType>;
    using pbat::sim::dynamics::EFemElastoDynamicsTimeStepInitialization;

    nb::enum_<EFemElastoDynamicsTimeStepInitialization>(
        m,
        "EFemElastoDynamicsTimeStepInitialization")
        .value("Position", EFemElastoDynamicsTimeStepInitialization::Position)
        .value("FreeTrajectory", EFemElastoDynamicsTimeStepInitialization::FreeTrajectory)
        .value(
            "TrajectoryWithExternalLoad",
            EFemElastoDynamicsTimeStepInitialization::TrajectoryWithExternalLoad)
        .value(
            "TrajectoryWithFdLoad",
            EFemElastoDynamicsTimeStepInitialization::TrajectoryWithFdLoad)
        .value(
            "TrajectoryWithProjectedFdLoad",
            EFemElastoDynamicsTimeStepInitialization::TrajectoryWithProjectedFdLoad)
        .export_values();

    nb::class_<ElastoDynamics>(m, "FemElastoDynamics")
        .def(nb::init<>(), "Construct an empty elasto-dynamics problem.")
        .def(
            "__init__",
            [](ElastoDynamics* self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> V,
               nb::DRef<Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic> const> C) {
                new (self) ElastoDynamics(V, C);
            },
            nb::arg("V"),
            nb::arg("C"),
            "Construct an elasto-dynamics problem on the mesh domain (V, C).\n\n"
            "All quantities are initialized to sensible defaults: rest pose positions, zero\n"
            "velocities, homogeneous rubber-like material properties, gravity load, and a\n"
            "BDF time integrator.\n\n"
            "Args:\n"
            "    V (numpy.ndarray): `kDims x |# verts|` matrix of mesh vertex positions.\n"
            "    C (numpy.ndarray): `|# cell nodes| x |# cells|` matrix of mesh cells.\n")
        .def_prop_rw(
            "X",
            [](ElastoDynamics const& self) { return self.mesh.X; },
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> X) {
                self.mesh.X = X;
            },
            "kDims x |# nodes| matrix of nodal coordinates")
        .def_prop_rw(
            "E",
            [](ElastoDynamics const& self) {
                return self.mesh.E; // |# elem nodes| x |#elements|
            },
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic> const> E) {
                self.mesh.E = E;
            },
            "|# elem nodes| x |# elements| matrix of element connectivity")
        .def_rw("x", &ElastoDynamics::x, "kDims x |# nodes| nodal positions")
        .def_rw("v", &ElastoDynamics::v, "kDims x |# nodes| nodal velocities")
        .def_rw("atfd", &ElastoDynamics::atfd, "kDims x |# nodes| finite-difference accelerations")
        .def_rw("fext", &ElastoDynamics::fext, "kDims x |# nodes| external forces at nodes")
        .def_rw("m", &ElastoDynamics::m, "|# nodes| x 1 lumped mass (per node)")
        .def_rw("xtilde", &ElastoDynamics::xtilde, "kDims x |# nodes| BDF inertial targets")
        .def_rw("bdf", &ElastoDynamics::bdf, "Underlying BDF time integrator")
        .def_rw(
            "wgU",
            &ElastoDynamics::wgU,
            "|# quad.pts.| x 1 quadrature weights for elastic potential")
        .def_rw(
            "GNegU",
            &ElastoDynamics::GNegU,
            "|ElementType::kNodes| x |kDims * # quad.pts.| shape function gradients at quadrature "
            "points")
        .def_rw(
            "lamegU",
            &ElastoDynamics::lamegU,
            "2 x |# quad.pts.| Lame coefficients at quadrature points")
        .def_rw(
            "UgU",
            &ElastoDynamics::UgU,
            "|# quad.pts.| x 1 elastic energy density at quadrature points")
        .def_rw(
            "GgU",
            &ElastoDynamics::GgU,
            "|# dims * # elem nodes| x |# quad.pts.| element elastic gradient vectors at "
            "quadrature points")
        .def_rw(
            "HgU",
            &ElastoDynamics::HgU,
            "|# dims * # elem nodes| x |# dims * # elem nodes * # quad.pts.| element elastic "
            "hessian matrices at quadrature points")
        .def_rw("ndbc", &ElastoDynamics::ndbc, "Number of Dirichlet constrained nodes")
        .def_rw(
            "dbc",
            &ElastoDynamics::dbc,
            "|# nodes| x 1 concatenated vector of Dirichlet unconstrained and constrained node "
            "indices, partitioned as (unconstrained |# nodes| x 1, constrained |# nodes| x 1)")
        .def_rw(
            "dmask",
            &ElastoDynamics::dmask,
            "`|# nodes| x 1` mask of Dirichlet boundary conditions s.t. `dmask(i) == true` if node "
            "i is constrained")
        .def(
            "M",
            [](ElastoDynamics const& self) { return self.M().eval(); },
            "kDims*|# nodes| x 1 vector of the lumped mass matrix diagonal (per dof)")
        .def(
            "aext",
            [](ElastoDynamics const& self) { return self.aext().eval(); },
            "kDims x |# nodes| external acceleration field")
        .def(
            "construct",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> V,
               nb::DRef<Eigen::Matrix<IndexType, Eigen::Dynamic, Eigen::Dynamic> const> C) {
                self.Construct(V, C);
            },
            nb::arg("V"),
            nb::arg("C"),
            "Construct the elasto-dynamics problem on the mesh domain (V, C). Returns self.\n\n"
            "Args:\n"
            "    V (numpy.ndarray): `kDims x |# verts|` matrix of mesh vertex positions.\n"
            "    C (numpy.ndarray): `|# cell nodes| x |# cells|` matrix of mesh cells.\n")
        .def(
            "set_initial_conditions",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> x0,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> v0) {
                self.SetInitialConditions(x0, v0);
            },
            nb::arg("x0"),
            nb::arg("v0"),
            "Set initial positions and velocities: x0, v0 as kDims x |# nodes| matrices.\n\n"
            "Args:\n"
            "    x0 (numpy.ndarray): `kDims x |# nodes|` matrix of initial positions.\n"
            "    v0 (numpy.ndarray): `kDims x |# nodes|` matrix of initial velocities.\n")
        .def(
            "set_mass_matrix",
            [](ElastoDynamics& self, ScalarType rho) { self.SetMassMatrix(rho); },
            nb::arg("rho"),
            "Compute, lump, and set the mass matrix with homogeneous density rho.\n\n"
            "Args:\n"
            "    rho (float): Mass density of the material.\n")
        .def(
            "set_mass_matrix",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> eg,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> wg,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> Xig,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> rhog) {
                self.SetMassMatrix(eg, wg, Xig, rhog);
            },
            nb::arg("eg"),
            nb::arg("wg"),
            nb::arg("Xig"),
            nb::arg("rhog"),
            "Compute and set the mass matrix with variable density rhog at quadrature points.\n\n"
            "Args:\n"
            "    eg (numpy.ndarray): `|# quadrature points| x 1` array of element indices for "
            "quadrature points.\n"
            "    wg (numpy.ndarray): `|# quadrature points| x 1` array of quadrature weights.\n"
            "    Xig (numpy.ndarray): `kDims x |# quadrature points|` matrix of quadrature point "
            "positions.\n"
            "    rhog (numpy.ndarray): `|# quadrature points| x 1` array of densities at "
            "quadrature points.\n")
        .def(
            "set_elastic_energy",
            [](ElastoDynamics& self, ScalarType mu, ScalarType lambda) {
                self.SetElasticEnergy(mu, lambda);
            },
            nb::arg("mu"),
            nb::arg("lambda"),
            "Set homogeneous elastic material (Lame parameters mu, lambda).\n\n"
            "Args:\n"
            "    mu (float): First Lame parameter.\n"
            "    lambda (float): Second Lame parameter.\n")
        .def(
            "set_elastic_energy",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> eg,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> wg,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> Xig,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> mug,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> lambdag) {
                self.SetElasticEnergy(eg, wg, Xig, mug, lambdag);
            },
            nb::arg("eg"),
            nb::arg("wg"),
            nb::arg("Xg"),
            nb::arg("mug"),
            nb::arg("lambdag"),
            "Set heterogeneous elastic material with per-quadrature Lame coefficients.\n\n"
            "Args:\n"
            "    eg (numpy.ndarray): `|# quadrature points| x 1` array of element indices for "
            "quadrature points.\n"
            "    wg (numpy.ndarray): `|# quadrature points| x 1` array of quadrature weights.\n"
            "    Xig (numpy.ndarray): `kDims x |# quadrature points|` matrix of quadrature point "
            "positions.\n"
            "    mug (numpy.ndarray): `|# quadrature points| x 1` array of first Lame parameters "
            "at quadrature points.\n"
            "    lambdag (numpy.ndarray): `|# quadrature points| x 1` array of second Lame "
            "parameters at quadrature points.\n")
        .def(
            "set_external_load",
            [](ElastoDynamics& self, nb::DRef<Eigen::Vector<ScalarType, kDims> const> b) {
                self.SetExternalLoad(b);
            },
            nb::arg("b"),
            "Set fixed body force (gravity, etc.) as kDims x 1 vector.\n\n"
            "Args:\n"
            "    b (numpy.ndarray): `kDims x 1` vector of body forces.\n")
        .def(
            "set_external_load",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Vector<IndexType, Eigen::Dynamic> const> eg,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> wg,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> Xig,
               nb::DRef<Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> const> bg) {
                self.SetExternalLoad(eg, wg, Xig, bg);
            },
            nb::arg("eg"),
            nb::arg("wg"),
            nb::arg("Xg"),
            nb::arg("bg"),
            "Set variable body forces bg at quadrature points.\n\n"
            "Args:\n"
            "    eg (numpy.ndarray): `|# quadrature points| x 1` array of element indices for "
            "quadrature points.\n"
            "    wg (numpy.ndarray): `|# quadrature points| x 1` array of quadrature weights.\n"
            "    Xig (numpy.ndarray): `kDims x |# quadrature points|` matrix of quadrature point "
            "positions.\n"
            "    bg (numpy.ndarray): `kDims x |# quadrature points|` matrix of body forces at "
            "quadrature points.\n")
        .def(
            "set_time_integration_scheme",
            [](ElastoDynamics& self, ScalarType dt, int s) {
                self.SetTimeIntegrationScheme(dt, s);
            },
            nb::arg("dt") = ScalarType(1e-2),
            nb::arg("s")  = 1,
            "Set BDF time integration scheme with time step dt and s-step order.\n\n"
            "Args:\n"
            "    dt (float, optional): Time step size. Defaults to 1e-2.\n"
            "    s (int, optional): BDF order (1 to 6). Defaults to 1.\n")
        .def(
            "constrain",
            [](ElastoDynamics& self, nb::DRef<Eigen::Vector<int, Eigen::Dynamic> const> D) {
                self.Constrain(D);
            },
            nb::arg("D"),
            "Set Dirichlet boundary conditions as |# nodes| integer mask.\n\n"
            "Args:\n"
            "    D (numpy.ndarray): `|#nodes|` integer array where non-zero values indicate a "
            "constrained node.\n")
        .def(
            "setup_time_integration_optimization",
            [](ElastoDynamics& self,
               EFemElastoDynamicsTimeStepInitialization eInitializationStrategy) {
                self.SetupTimeIntegrationOptimization(eInitializationStrategy);
            },
            nb::arg("initialization_strategy") = EFemElastoDynamicsTimeStepInitialization::Position,
            "Compute BDF inertial target for implicit time stepping (updates xtilde).\n\n"
            "Args:\n"
            "    initialization_strategy (EFemElastoDynamicsTimeStepInitialization): Strategy for "
            "initializing the time step (i.e. the time integration optimization problem's initial "
            "iterate). Defaults to Position.\n")
        .def(
            "step",
            &ElastoDynamics::Step,
            "Perform a single time integration step using `x`, `v`.")
        .def(
            "back_substitute_integrated_positions_into_velocities",
            &ElastoDynamics::BackSubstituteIntegratedPositionsIntoVelocities,
            "Update velocities after a position-based time integration solve.")
        .def(
            "compute_elastic_energy",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> x,
               fem::EElementElasticityComputationFlags compute_flags,
               fem::EHyperElasticSpdCorrection spd_correction) {
                self.ComputeElasticEnergy(x.reshaped(), compute_flags, spd_correction);
            },
            nb::arg("x"),
            nb::arg("compute_flags"),
            nb::arg("spd_correction"),
            "Compute per-quadrature elastic energy, gradient, and/or Hessian with optional SPD\n"
            "correction.\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `kDims*|# nodes| x 1` vector of nodal positions.\n"
            "    compute_flags (ElementElasticityComputationFlags): Bitmask flags indicating which "
            "quantities to compute.\n"
            "    spd_correction (HyperElasticSpdCorrection): Hessian SPD correction strategy.\n")
        .def(
            "objective",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> x) {
                return self.Objective(x.reshaped());
            },
            nb::arg("x"),
            "Compute the time integration optimization's objective function value.\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `kDims*|# nodes| x 1` vector of nodal positions.\n\n"
            "Returns:\n"
            "    float: Objective function value.")
        .def(
            "objective",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> x) {
                return self.Objective(x);
            },
            nb::arg("x"),
            "Compute the time integration optimization's objective function value.\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `kDims*|# nodes| x 1` vector of nodal positions.\n\n"
            "Returns:\n"
            "    float: Objective function value.")
        .def(
            "gradient",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> const> x) {
                return self.Gradient(x.reshaped());
            },
            nb::arg("x"),
            "Compute the time integration optimization's gradient.\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `kDims*|# nodes| x 1` vector of nodal positions.\n\n"
            "Returns:\n"
            "    numpy.ndarray: kDims*|# nodes| gradient vector.")
        .def(
            "gradient",
            [](ElastoDynamics& self,
               nb::DRef<Eigen::Vector<ScalarType, Eigen::Dynamic> const> x) {
                return self.Gradient(x);
            },
            nb::arg("x"),
            "Compute the time integration optimization's gradient.\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `kDims*|# nodes| x 1` vector of nodal positions.\n\n"
            "Returns:\n"
            "    numpy.ndarray: kDims*|# nodes| gradient vector.")
        .def("is_dirichlet_node", &ElastoDynamics::IsDirichletNode, nb::arg("node"))
        .def("is_dirichlet_dof", &ElastoDynamics::IsDirichletDof, nb::arg("i"))
        .def_prop_ro(
            "dirichlet_nodes",
            [](ElastoDynamics const& self) { return self.DirichletNodes().eval(); },
            "ndbc x 1 array of Dirichlet constrained node indices")
        .def_prop_ro(
            "dirichlet_dofs",
            [](ElastoDynamics const& self) { return self.DirichletDofs().eval(); },
            "kDims*ndbc x 1 array of Dirichlet constrained dofs")
        .def_prop_ro(
            "dirichlet_coordinates",
            [](ElastoDynamics const& self) { return self.DirichletCoordinates().eval(); },
            "kDims x ndbc Dirichlet constrained nodal coordinates")
        .def_prop_ro(
            "dirichlet_velocities",
            [](ElastoDynamics const& self) { return self.DirichletVelocities().eval(); },
            "kDims x ndbc Dirichlet constrained nodal velocities")
        .def_prop_ro(
            "free_nodes",
            [](ElastoDynamics const& self) { return self.FreeNodes().eval(); },
            "|# nodes|-ndbc array of unconstrained node indices")
        .def_prop_ro(
            "free_dofs",
            [](ElastoDynamics const& self) { return self.FreeDofs().eval(); },
            "|kDims*# nodes - ndbc| array of unconstrained dofs")
        .def_prop_ro(
            "free_coordinates",
            [](ElastoDynamics const& self) { return self.FreeCoordinates().eval(); },
            "kDims x (|#nodes|-ndbc) unconstrained nodal coordinates")
        .def_prop_ro(
            "free_velocities",
            [](ElastoDynamics const& self) { return self.FreeVelocities().eval(); },
            "kDims x (|#nodes|-ndbc) unconstrained nodal velocities")
        .def(
            "serialize",
            &ElastoDynamics::Serialize,
            nb::arg("archive"),
            "Serialize the problem to an archive")
        .def(
            "deserialize",
            &ElastoDynamics::Deserialize,
            nb::arg("archive"),
            "Deserialize the problem from an archive")
        // Class docstring
        .doc() = R"doc(
Finite Element Elasto-Dynamics with BDF time integration.

This class models a finite element elasto-dynamics initial value problem with Dirichlet
boundary conditions, using a backward differentiation formula (BDF) time discretization.
The dynamics at each step can be expressed as the minimization of a quadratic inertia term
around a BDF target plus the hyperelastic potential energy of the configuration.

The Python bindings only support 3D linear tetrahedral meshes with stable neo-Hookean materials. 
Use C++ for other element types, dimensions and orders.

Typical workflow:
```
    fem = FemElastoDynamics(V, C)
    fem.set_mass_matrix(rho)
    fem.set_elastic_energy(mu, lambda)
    fem.set_external_load(b)
    fem.set_time_integration_scheme(dt=1e-2, s=1)
    fem.constrain(D)  # int mask of size |#nodes|
    fem.setup_time_integration_optimization()  # compute inertial target xtilde
```
You can then assemble equations with compute_elastic_energy(...) and use the returned
quantities together with fem.bdf to build and solve your implicit step.
)doc";
}

} // namespace pbat::py::sim::dynamics