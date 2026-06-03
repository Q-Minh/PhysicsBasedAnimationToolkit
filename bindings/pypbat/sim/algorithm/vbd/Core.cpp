#include "Core.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/physics/Enums.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pbat/sim/algorithm/vbd/Core.h>
#include <pbat/sim/algorithm/vbd/Enums.h>
#include <pbat/sim/contact/MeshDynamics.h>
#include <pbat/sim/dynamics/FemElastoDynamics.h>
#include <tuple>

namespace pbat::py::sim::algorithm::vbd {

void BindCore(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using pbat::sim::algorithm::common::FemElastoDynamics;
    using pbat::sim::algorithm::vbd::ESALPenaltyStiffness;
    using pbat::sim::algorithm::vbd::EStencilGradientBetaWarmStartMask;
    using pbat::sim::algorithm::vbd::EVertexIntegrationLinearSolver;
    using pbat::sim::algorithm::vbd::Params;

    nb::enum_<EVertexIntegrationLinearSolver>(m, "VertexIntegrationLinearSolver")
        .value("Inverse", EVertexIntegrationLinearSolver::Inverse)
        .value("LLT", EVertexIntegrationLinearSolver::LLT)
        .value("QR", EVertexIntegrationLinearSolver::QR)
        .value("EVD", EVertexIntegrationLinearSolver::EVD)
        .export_values();

    nb::enum_<ESALPenaltyStiffness>(m, "SALPenaltyStiffness")
        .value("LocalMaxRayleighQuotient", ESALPenaltyStiffness::LocalMaxRayleighQuotient)
        .value("GlobalMaxRayleighQuotient", ESALPenaltyStiffness::GlobalMaxRayleighQuotient)
        .export_values();

    nb::enum_<EStencilGradientBetaWarmStartMask>(m, "StencilGradientBetaWarmStartMask")
        .value("Never", EStencilGradientBetaWarmStartMask::None)
        .value("Subproblem", EStencilGradientBetaWarmStartMask::Subproblem)
        .value("TimeStep", EStencilGradientBetaWarmStartMask::TimeStep)
        .export_values();

    m.def(
        "vertex_element_adjacency_graph",
        [](nb::DRef<pbat::IndexMatrixX const> const& E, Index nNodes) {
            IndexVectorX GVGp(nNodes + 1);
            IndexVectorX GVGe(E.size());
            IndexVectorX GVGilocal(E.size());
            pbat::sim::algorithm::vbd::VertexElementAdjacencyGraph(
                E,
                nNodes,
                GVGp,
                GVGe,
                GVGilocal);
            return std::make_tuple(GVGp, GVGe, GVGilocal);
        },
        nb::arg("E"),
        nb::arg("n_nodes"),
        "Compute the vertex-element adjacency graph.\n\n"
        "Args:\n"
        "    elements (numpy.ndarray): `|# elems| x |elem dim|` element connectivity\n"
        "Returns:\n"
        "    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A 3-tuple containing:\n"
        "    GVGp (numpy.ndarray): `|# verts + 1|` prefixes into GVGe\n"
        "    GVGe (numpy.ndarray): `|# of vertex-elems adjacencies|` element indices s.t. "
        "`GVGe[k] for GVGp[i] <= k < GVGp[i+1]` gives the element `e` adjacent to vertex `i`\n"
        "    GVGilocal (numpy.ndarray): `|# of vertex-elems adjacencies|` local vertex indices "
        "s.t. `GVGilocal[k] for GVGp[i] <= k < GVGp[i+1]` gives the local vertex index of "
        "vertex `i` in element `e=GVGe[k]`");

    m.def(
        "vertex_colors",
        [](nb::DRef<pbat::IndexMatrixX const> const& E,
           Index nNodes,
           graph::EGreedyColorOrderingStrategy eOrdering,
           graph::EGreedyColorSelectionStrategy eSelection) {
            IndexVectorX colors(nNodes);
            IndexVectorX GVVp(nNodes + 1);
            IndexVectorX GVVadj{};
            pbat::sim::algorithm::vbd::VertexColors(
                E,
                nNodes,
                eOrdering,
                eSelection,
                GVVp,
                GVVadj,
                colors);
            return std::make_tuple(GVVp, GVVadj, colors);
        },
        nb::arg("E"),
        nb::arg("n_nodes"),
        nb::arg("ordering")  = graph::EGreedyColorOrderingStrategy::LargestDegree,
        nb::arg("selection") = graph::EGreedyColorSelectionStrategy::LeastUsed,
        "Compute vertex colors using a greedy algorithm.\n\n"
        "Args:\n"
        "    elements (numpy.ndarray): `|# elems| x |elem dim|` element connectivity\n"
        "    n_nodes (int): Number of nodes in the mesh\n"
        "    ordering (pbat.graph.EGreedyColorOrderingStrategy): Vertex color ordering strategy\n"
        "    selection (pbat.graph.EGreedyColorSelectionStrategy): Vertex color selection "
        "strategy\n"
        "Returns:\n"
        "    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: (GVVp, GVVadj, colors) where "
        "`GVVp` is a `|# verts + 1| x 1` array of pointers into `GVVadj`, `GVVadj` is a `|# "
        "vertex-vertex adjacencies| x 1` array of adjacent vertex indices, and `colors` is a `|# "
        "verts| x 1` array of vertex colors");

    nb::class_<Params>(m, "Params")
        .def(nb::init<>())
        .def(
            "with_vertex_element_adjacency_graph",
            &Params::WithVertexElementAdjacencyGraph,
            nb::arg("GVGp"),
            nb::arg("GVGe"),
            nb::arg("GVGilocal"),
            nb::rv_policy::reference_internal,
            "Vertex-element adjacency graph.\n\n"
            "Args:\n"
            "    GVGp (numpy.ndarray): `|# verts + 1|` prefixes into GVGe\n"
            "    GVGe (numpy.ndarray): `|# of vertex-elems adjacencies|` element indices s.t. "
            "`GVGe[k] for GVGp[i] <= k < GVGp[i+1]` gives the element `e` adjacent to vertex `i`\n"
            "    GVGilocal (numpy.ndarray): `|# of vertex-elems adjacencies|` local vertex indices "
            "s.t. `GVGilocal[k] for GVGp[i] <= k < GVGp[i+1]` gives the local vertex index of "
            "vertex `i` in element `e=GVGe[k]`\n"
            "Returns:\n"
            "    self (Params): Reference to this")
        .def(
            "with_vertex_colors",
            &Params::WithVertexColors,
            nb::arg("GVVp"),
            nb::arg("GVVadj"),
            nb::arg("colors"),
            nb::rv_policy::reference_internal,
            "Vertex colors used for coloring the VBD solve.\n\n"
            "Args:\n"
            "    colors (numpy.ndarray): `|# verts|` vertex colors\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_damping",
            &Params::WithDamping,
            nb::arg("betaR"),
            nb::rv_policy::reference_internal,
            "Rayleigh damping coefficient.\n\n"
            "Args:\n"
            "    betaR (float): Rayleigh damping coefficient\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_maximum_iterations",
            &Params::WithMaximumIterations,
            nb::arg("n_iters"),
            nb::rv_policy::reference_internal,
            "Maximum number of outer iterations.\n\n"
            "Args:\n"
            "    n_iters (int): Maximum number of outer iterations\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_subproblem_maximum_iterations",
            &Params::WithSubproblemMaximumIterations,
            nb::arg("n_iters"),
            nb::rv_policy::reference_internal,
            "Maximum number of VBD iterations per subproblem.\n\n"
            "Args:\n"
            "    n_iters (int): Maximum number of VBD iterations per subproblem\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_stencil_gradient_acceleration",
            &Params::WithStencilGradientAcceleration,
            nb::arg("betaG0")                                  = Scalar(0.5),
            nb::arg("rhohat")                                  = Scalar(0.005),
            nb::arg("gammadown")                               = Scalar(0.5),
            nb::arg("gammaup")                                 = Scalar(0.5),
            nb::arg("rhohatS")                                 = Scalar(0.005),
            nb::arg("gammadownS")                              = Scalar(0.5),
            nb::arg("gammaupS")                                = Scalar(0.5),
            nb::arg("surface_stencil_surface_neighbours_only") = true,
            nb::arg("warm_start_beta") = EStencilGradientBetaWarmStartMask::Subproblem,
            nb::rv_policy::reference_internal,
            "Stencil gradient acceleration parameters.\n\n"
            "Args:\n"
            "    betaG0 (float): Initial stencil gradient augmentation coefficient\n"
            "    rhohat (float): Stencil gradient density factor\n"
            "    gammadown (float): Stencil gradient beta reduction factor\n"
            "    gammaup (float): Stencil gradient beta increase factor\n"
            "    rhohatS (float): Stencil gradient density factor (surface)\n"
            "    gammadownS (float): Stencil gradient beta reduction factor (surface)\n"
            "    gammaupS (float): Stencil gradient beta increase factor (surface)\n"
            "    surface_stencil_surface_neighbours_only (bool): If true, only consider surface "
            "nodes in stencil gradient acceleration\n"
            "    warm_start_beta (StencilGradientBetaWarmStartMask): If Subproblem, initialize "
            "beta for the first iteration of each subproblem to the final beta from the previous "
            "subproblem (default: Subproblem)\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_vertex_linear_solver",
            &Params::WithVertexLinearSolver,
            nb::arg("solver"),
            nb::arg("zero")  = std::numeric_limits<Scalar>::epsilon(),
            nb::arg("eps")   = std::numeric_limits<Scalar>::epsilon(),
            nb::arg("iters") = -1,
            nb::rv_policy::reference_internal,
            "Numerical zero for hessian singularity check.\n\n"
            "Args:\n"
            "    solver (pbat.sim.algorithm.vbd.EVertexIntegrationLinearSolver): Vertex "
            "integration linear solver\n"
            "    zero (float): Numerical zero\n"
            "    eps (float): Vertex integration linear solver epsilon\n"
            "    iters (int): Maximum number of vertex integration linear solver iterations\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "construct",
            &Params::Construct,
            nb::arg("validate") = true,
            nb::rv_policy::reference_internal,
            "Construct the Params object.\n\n"
            "Args:\n"
            "    validate (bool): Throw on detected ill-formed inputs\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "serialize",
            &Params::Serialize,
            nb::arg("archive"),
            nb::arg("minimal") = true,
            "Serialize this to archive.\n\n"
            "Args:\n"
            "    archive: Archive to serialize to\n"
            "    minimal (bool): If True (default), only serialize stateless configuration "
            "parameters. If False, also serialize solver state and mesh-dependent data.")
        .def(
            "deserialize",
            &Params::Deserialize,
            nb::arg("archive"),
            "Deserialize this from archive.")
        .def_rw("GVGp", &Params::GVGp, "`|# verts+1|` prefixes into GVGe")
        .def_rw("GVGe", &Params::GVGe, "`|# of vertex-elems adjacencies|` element indices")
        .def_rw(
            "GVGilocal",
            &Params::GVGilocal,
            "`|# of vertex-elems adjacencies|` local vertex indices")
        .def_rw("GVVp", &Params::GVVp, "`|# verts+1|` prefixes into GVVadj")
        .def_rw(
            "GVVadj",
            &Params::GVVadj,
            "`|# vertex-vertex adjacencies|` adjacent vertex indices")
        .def_rw("colors", &Params::colors, "`|# verts|` vertex colors")
        .def_rw(
            "Pptr",
            &Params::Pptr,
            "`|# partitions+1|` partition pointers, s.t. the range `[Pptr[p], Pptr[p+1])` indexes "
            "into Padj from partition `p`")
        .def_rw("Padj", &Params::Padj, "`|# verts|` partition vertices")
        .def_rw("betaR", &Params::betaR, "Rayleigh damping coefficient")
        .def_rw("n_max_iters", &Params::nMaxIters, "Maximum number of outer iterations")
        .def_rw(
            "n_subproblem_max_iters",
            &Params::nSubproblemMaxIters,
            "Maximum number of VBD iterations per subproblem")
        .def_rw("gtol", &Params::gtol, "Gradient norm convergence threshold")
        .def_rw(
            "e_penalty_stiffness",
            &Params::ePenaltyStiffness,
            "Strategy for updating the augmented Lagrangian penalty parameter")
        .def_rw("vlinsolve", &Params::eSolver, "Vertex integration linear solver")
        .def_rw("hess_zero", &Params::hessZero, "Determinant of Hessian zero threshold")
        .def_rw("vls_eps", &Params::vLinSolverEps, "Vertex integration linear solver epsilon")
        .def_rw(
            "vls_max_iters",
            &Params::vLinSolverMaxIters,
            "Maximum number of vertex integration linear solver iterations")
        .def_rw("betaG", &Params::betaG, "Per-vertex stencil gradient augmentation scale")
        .def_rw("betaG0", &Params::betaG0, "Initial stencil gradient augmentation scale")
        .def_rw(
            "rhohat",
            &Params::rhohat,
            "Lipschitz-normalized threshold for considering steps small")
        .def_rw("gammadown", &Params::gammadown, "Beta reduction factor")
        .def_rw("gammaup", &Params::gammaup, "Beta increase factor")
        .def_rw(
            "rhohatS",
            &Params::rhohatS,
            "Lipschitz-normalized threshold for considering steps small (surface)")
        .def_rw("gammadownS", &Params::gammadownS, "Beta reduction factor (surface)")
        .def_rw("gammaupS", &Params::gammaupS, "Beta increase factor (surface)")
        .def_rw(
            "surface_stencil_surface_neighbours_only",
            &Params::bSurfaceStencilSurfaceNeighboursOnly,
            "Whether to only consider surface neighbors for surface nodes in stencil gradient "
            "acceleration")
        .def_rw(
            "warm_start_beta",
            &Params::eWarmStartMask,
            "Warm start mask for stencil gradient augmentation scale initialization")
        .def_rw("k", &Params::k, "Current iteration")
        .def_rw("kp", &Params::kp, "Current subproblem iteration");

    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using MeshDynamicsType  = pbat::sim::contact::MeshDynamics<ScalarType, IndexType>;

    m.def(
        "initialize_solve",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            pbat::sim::algorithm::vbd::InitializeSolve<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Computes OGC query radius, updates the constraint set, restores feasibility,\n"
        "and resets solver state.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n");
    m.def(
        "linearize_constraints",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact) {
            pbat::sim::algorithm::vbd::LinearizeConstraints<ElasticEnergyType>(fem, contact);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        "Linearize contact constraints at the current iterate.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n");
    m.def(
        "check_convergence",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType const& contact,
           Params& params) {
            return pbat::sim::algorithm::vbd::CheckConvergence<ElasticEnergyType>(
                fem,
                contact,
                params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Check convergence of VBD solve.\n\n"
        "Computes the full gradient (elastic + momentum + contact) and checks if its norm\n"
        "is below the convergence threshold params.gtol.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n\n"
        "Returns:\n"
        "    bool: True if converged, False otherwise.");
    m.def(
        "prepare_subproblem",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            pbat::sim::algorithm::vbd::PrepareSubproblem<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Prepare a linearized constraint subproblem.\n\n"
        "Assembles the block-diagonal dynamics Hessian, updates the penalty parameter,\n"
        "and optionally resets the stencil gradient acceleration coefficients.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n");
    m.def(
        "iterate",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            pbat::sim::algorithm::vbd::Iterate<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Perform one alternate slack and position update on the augmented Lagrangian subproblem "
        "using VBD.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n");
    m.def(
        "finalize_subproblem",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            pbat::sim::algorithm::vbd::FinalizeSubproblem<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Finalize the current linearized constraint subproblem.\n\n"
        "Updates dual variables (slack, Lagrange multiplier, decay), restores feasibility,\n"
        "and updates the constraint set for the next subproblem.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n");
    m.def(
        "solve",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            return pbat::sim::algorithm::vbd::Solve<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Solve the VBD minimization.\n\n"
        "High-level convenience function. The equivalent low-level loop is:\n"
        "  initialize_solve(fem, contact, params)\n"
        "  for params.k in range(params.n_max_iters):\n"
        "      linearize_constraints(fem, contact)\n"
        "      if check_convergence(fem, contact, params): break\n"
        "      prepare_subproblem(fem, contact, params)\n"
        "      params.kp = 0\n"
        "      while params.kp < params.n_subproblem_max_iters:\n"
        "          iterate(fem, contact, params)\n"
        "      finalize_subproblem(fem, contact, params)\n"
        "  fem.back_substitute_integrated_positions_into_velocities()\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "Returns:\n"
        "    bool: True if the solver converged, False otherwise\n");
    m.def(
        "integrate",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            pbat::sim::algorithm::vbd::Integrate<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Integrate one time step using VBD as non-linear solver.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters");
}

} // namespace pbat::py::sim::algorithm::vbd
