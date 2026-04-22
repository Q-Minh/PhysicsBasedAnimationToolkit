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
        .def("serialize", &Params::Serialize, nb::arg("archive"), "Serialize this to archive.")
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
            "Serialize parameters to an archive")
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
        .def_ro("k", &Params::k, "Current outer iteration index");

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
        "Computes the full gradient and checks if the squared gradient norm is below\n"
        "the convergence threshold params.newton.gtol2.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n\n"
        "Returns:\n"
        "    bool: True if KKT conditions are satisfied (converged), False otherwise.");
    m.def(
        "prepare_subproblem",
        [](ElastoDynamics& fem, MeshDynamics& contact, Params& params) {
            pbat::sim::algorithm::newton::PrepareSubproblem(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Prepare a linearized constraint subproblem.\n\n"
        "Precomputes elastic energy derivatives, updates barrier parameters, and initializes the "
        "inner Newton solver.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n");
    m.def(
        "prepare_next_iteration",
        [](ElastoDynamics& fem,
           MeshDynamics& contact,
           Params& params,
           bool bAreSubproblemDerivativesDirty) {
            pbat::sim::algorithm::newton::PrepareNextIteration(
                fem,
                contact,
                params,
                bAreSubproblemDerivativesDirty);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        nb::arg("are_subproblem_derivatives_dirty") = true,
        "Prepare next iteration of the current linearized constraint subproblem.\n\n"
        "Recomputes elastic energy derivatives and evaluates the merit function and gradient\n"
        "for the inner Newton solver. Call this after each iterate() within a subproblem.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    contact (MeshDynamics): Contact dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n"
        "    bAreSubproblemDerivativesDirty (bool): Whether to compute subproblem "
        "derivatives.\n\n");
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
        "Restores feasibility, updates the constraint set, and increments the outer\n"
        "iteration counter params.k.\n\n"
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
        "  while params.k < params.n_max_iters:\n"
        "      linearize_constraints(fem, contact)\n"
        "      if check_convergence(fem, contact, params): break\n"
        "      prepare_subproblem(fem, contact, params)\n"
        "      prepare_next_iteration(fem, contact, params)\n"
        "      while params.newton.k < params.newton.n_max_iters:\n"
        "          if params.newton.gknorm2 <= params.newton.gtol2: break\n"
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
