#include "Core.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <pbat/Aliases.h>
#include <pbat/fem/HyperElasticPotential.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pbat/sim/algorithm/newton/Core.h>

namespace pbat::py::sim::algorithm::newton {

void BindCore(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::sim::algorithm::newton::ELinearSolver;
    using pbat::sim::algorithm::newton::EOgcTruncationStrategy;
    using pbat::sim::algorithm::newton::Params;
    using ScalarType = pbat::Scalar;
    using IndexType  = pbat::Index;

    nb::enum_<ELinearSolver>(m, "ELinearSolver")
        .value("LLT", ELinearSolver::LLT, "Cholesky LLT decomposition")
        .value(
            "PCGJacobi",
            ELinearSolver::PCGJacobi,
            "Preconditioned Conjugate Gradient with Jacobi (i.e. diagonal) preconditioner")
        .value(
            "PCGIC",
            ELinearSolver::PCGIC,
            "Preconditioned Conjugate Gradient with Incomplete Cholesky preconditioner")
        .value(
            "PCGILUT",
            ELinearSolver::PCGILUT,
            "Preconditioned Conjugate Gradient with Incomplete LU with thresholding preconditioner")
        .value(
            "PCGLaplacian",
            ELinearSolver::PCGLaplacian,
            "Preconditioned Conjugate Gradient with Laplacian preconditioner")
        .export_values();

    nb::enum_<EOgcTruncationStrategy>(m, "EOgcTruncationStrategy")
        .value("PerVertex", EOgcTruncationStrategy::PerVertex, "Per-vertex OGC truncation")
        .value("Global", EOgcTruncationStrategy::Global, "Global OGC truncation")
        .export_values();

    nb::class_<Params>(m, "Params")
        .def(nb::init<>(), "Newton solver parameters and buffers.")
        .def(
            "serialize",
            &Params::Serialize,
            nb::arg("archive"),
            nb::arg("minimal") = true,
            "Serialize this to archive.\n\n"
            "Args:\n"
            "    archive: Archive to serialize to\n"
            "    minimal (bool): If True (default), only serialize stateless configuration "
            "parameters. If False, also serialize the Newton optimizer state.")
        .def(
            "deserialize",
            &Params::Deserialize,
            nb::arg("archive"),
            "Deserialize this from archive.")
        .def_rw("newton", &Params::newton, "Underlying Newton optimizer (math.optimization.Newton)")
        .def_rw(
            "hessian",
            &Params::hessian,
            "Sparse Hessian matrix (Eigen::SparseMatrix in CSC format)")
        .def_rw(
            "spd_correction",
            &Params::eSpdCorrection,
            "HyperElasticSpdCorrection used for SPD correction of elastic Hessians")
        .def(
            "with_optimizer",
            &Params::WithOptimizer,
            nb::arg("optimizer"),
            nb::rv_policy::reference_internal,
            "Set the underlying Newton optimizer. Returns self.")
        .def(
            "with_spd_correction",
            &Params::WithSpdCorrection,
            nb::arg("spd_correction"),
            nb::rv_policy::reference_internal,
            "Set the SPD correction mode for hyper-elastic Hessians. Returns self.")
        .def(
            "with_linear_solver",
            &Params::WithLinearSolver,
            nb::arg("linear_solver") = ELinearSolver::LLT,
            nb::arg("max_iters")     = 100,
            nb::arg("tol")           = ScalarType(1e-6),
            nb::rv_policy::reference_internal,
            "Set the linear solver type for the Newton step. Returns self.")
        .def(
            "with_ogc_truncation_strategy",
            &Params::WithOgcTruncationStrategy,
            nb::arg("strategy"),
            nb::rv_policy::reference_internal,
            "Set the OGC truncation strategy. Returns self.")
        .def(
            "with_max_iters",
            &Params::WithMaxIters,
            nb::arg("n"),
            nb::rv_policy::reference_internal,
            "Set the maximum number of outer (linearized constraint subproblem) iterations. "
            "Returns self.")
        .def(
            "construct",
            &Params::Construct,
            nb::arg("validate") = true,
            nb::rv_policy::reference_internal,
            "Construct the parameter set (optionally validating inputs). Returns self.")
        .def(
            "serialize",
            &Params::Serialize,
            nb::arg("archive"),
            nb::arg("minimal") = true,
            "Serialize parameters to an archive.\n\n"
            "Args:\n"
            "    archive: Archive to serialize to\n"
            "    minimal (bool): If True (default), only serialize stateless configuration "
            "parameters. If False, also serialize the Newton optimizer state.")
        .def(
            "deserialize",
            &Params::Deserialize,
            nb::arg("archive"),
            "Deserialize parameters from an archive")
        .def_prop_ro(
            "triplets",
            [](Params const& self) {
                Eigen::Vector<pbat::Index, Eigen::Dynamic> rows(self.triplets.size());
                Eigen::Vector<pbat::Index, Eigen::Dynamic> cols(self.triplets.size());
                Eigen::Vector<pbat::Scalar, Eigen::Dynamic> vals(self.triplets.size());
                for (size_t i = 0; i < self.triplets.size(); ++i)
                {
                    rows(static_cast<pbat::Index>(i)) = self.triplets[i].row();
                    cols(static_cast<pbat::Index>(i)) = self.triplets[i].col();
                    vals(static_cast<pbat::Index>(i)) = self.triplets[i].value();
                }
                return std::make_tuple(rows, cols, vals);
            },
            "Hessian triplets as (rows, cols, vals) arrays.")
        .def_rw("linear_solver", &Params::eLinearSolver, "Linear solver type used for Newton step")
        .def_rw(
            "ogc_truncation_strategy",
            &Params::eOgcTruncationStrategy,
            "OGC truncation strategy used during Newton iterations")
        .def_rw(
            "n_max_iters",
            &Params::nMaxIters,
            "Maximum number of outer (linearized constraint subproblem) iterations")
        .def_rw("k", &Params::k, "Current outer iteration index");

    // Bind algorithm functions for a concrete energy model (3D stable neo-Hookean)
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using ElastoDynamics    = pbat::sim::algorithm::newton::FemElastoDynamics<ElasticEnergyType>;
    using MeshDynamics      = pbat::sim::algorithm::newton::MeshDynamics;

    m.def(
        "initialize_solve",
        [](ElastoDynamics& fem, MeshDynamics& contact, Params& params) {
            pbat::sim::algorithm::newton::InitializeSolve<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Initialize a time-step solve.\n\n"
        "Computes OGC query radius, updates the constraint set, restores feasibility, and\n"
        "resets the outer iteration counter params.k = 0.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n");
    m.def(
        "linearize_constraints",
        [](ElastoDynamics& fem, MeshDynamics& contact) {
            pbat::sim::algorithm::newton::LinearizeConstraints(fem, contact);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        "Linearize constraints at the current iterate.\n\n"
        "Calls contact.linearize_constraints(x) to compute the linearized constraint data\n"
        "(chat, gradc) for the current positions.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n");
    m.def(
        "check_convergence",
        [](ElastoDynamics& fem, MeshDynamics& contact, Params& params) {
            return pbat::sim::algorithm::newton::CheckConvergence(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Check KKT convergence of the outer (nonlinear) problem.\n\n"
        "Precomputes elastic energy derivatives, then computes the full gradient and\n"
        "checks if the squared gradient norm is below the convergence threshold\n"
        "params.newton.gtol2.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n\n"
        "Returns:\n"
        "    bool: True if KKT conditions are satisfied (converged), False otherwise.");
    m.def(
        "prepare_subproblem",
        [](ElastoDynamics& fem,
           MeshDynamics& contact,
           Params& params,
           bool bAssumePostConvergenceCheck) {
            pbat::sim::algorithm::newton::PrepareSubproblem(
                fem,
                contact,
                params,
                bAssumePostConvergenceCheck);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        nb::arg("assume_post_convergence_check") = true,
        "Prepare a linearized constraint subproblem.\n\n"
        "Assembles the Hessian (without contact contributions), updates barrier parameters,\n"
        "and initializes the inner Newton solver. If assume_post_convergence_check is True,\n"
        "assumes that check_convergence() has already computed elastic derivatives.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n"
        "    assume_post_convergence_check (bool): If True (default), skips elastic\n"
        "        derivative computation (assumes check_convergence was called prior).\n");
    m.def(
        "prepare_next_iteration",
        [](ElastoDynamics& fem,
           MeshDynamics& contact,
           Params& params,
           bool bAssumePostConvergenceCheck) {
            pbat::sim::algorithm::newton::PrepareNextIteration(
                fem,
                contact,
                params,
                bAssumePostConvergenceCheck);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        nb::arg("assume_post_convergence_check") = true,
        "Prepare next iteration of the current linearized constraint subproblem.\n\n"
        "Evaluates the merit function and gradient for the inner Newton solver.\n"
        "If assume_post_convergence_check is False, also recomputes elastic energy\n"
        "derivatives before evaluating the merit function.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n"
        "    assume_post_convergence_check (bool): If True (default), skips elastic\n"
        "        derivative computation. Set to False after iterate() moves positions.\n");
    m.def(
        "iterate",
        [](ElastoDynamics& fem, MeshDynamics& contact, Params& params) {
            return pbat::sim::algorithm::newton::Iterate(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Perform one Newton iteration of the current linearized constraint subproblem.\n\n"
        "Assembles the Hessian (with linearized contact contributions), computes the search\n"
        "direction, and takes a Newton step with line search.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n\n"
        "Returns:\n"
        "    bool: True if a step was taken, False otherwise.");
    m.def(
        "finalize_subproblem",
        [](ElastoDynamics& fem, MeshDynamics& contact, Params& params) {
            pbat::sim::algorithm::newton::FinalizeSubproblem(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Finalize the current linearized constraint subproblem.\n\n"
        "Updates dual variables (slack, Lagrange multiplier, decay), restores feasibility, and "
        "updates the constraint set.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n");
    m.def(
        "solve",
        [](ElastoDynamics& fem, MeshDynamics& contact, Params& params) {
            return pbat::sim::algorithm::newton::Solve(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Run the full Newton solver to convergence (or until max iterations).\n\n"
        "High-level convenience function. The equivalent low-level loop is:\n"
        "  initialize_solve(fem, contact, params)\n"
        "  for params.k in range(params.n_max_iters):\n"
        "      linearize_constraints(fem, contact)\n"
        "      converged = check_convergence(fem, contact, params)\n"
        "      if converged: break\n"
        "      prepare_subproblem(fem, contact, params)\n"
        "      prepare_next_iteration(fem, contact, params, True)\n"
        "      while not converged:\n"
        "          if not iterate(fem, contact, params): break\n"
        "          prepare_next_iteration(fem, contact, params)\n"
        "      finalize_subproblem(fem, contact, params)\n"
        "  fem.back_substitute_integrated_positions_into_velocities()\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n\n"
        "Returns:\n"
        "    bool: True if convergence is achieved, False otherwise.");
}

} // namespace pbat::py::sim::algorithm::newton
