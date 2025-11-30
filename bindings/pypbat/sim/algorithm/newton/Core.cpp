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

    nb::class_<Params>(m, "Params")
        .def(nb::init<>(), "Newton solver parameters and buffers.")
        .def("serialize", &Params::Serialize, nb::arg("archive"), "Serialize this to archive.")
        .def(
            "deserialize",
            &Params::Deserialize,
            nb::arg("archive"),
            "Deserialize this from archive.")
        .def_rw("newton", &Params::newton, "Underlying Newton optimizer (math.optimization.Newton)")
        .def_ro(
            "ordering",
            &Params::ordering,
            "Triplet ordering for sparse Hessian assembly (|# triplets| x 1 integer array)")
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
        .def_rw("linear_solver", &Params::eLinearSolver, "Linear solver type used for Newton step");

    // Bind algorithm functions for a concrete energy model (3D stable neo-Hookean)
    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using ElastoDynamics    = pbat::sim::algorithm::newton::FemElastoDynamics<ElasticEnergyType>;

    m.def(
        "prepare_next_iteration",
        [](ElastoDynamics& fem, Params& params) {
            return pbat::sim::algorithm::newton::PrepareNextIteration<ElasticEnergyType>(
                fem,
                params);
        },
        nb::arg("fem"),
        nb::arg("params"),
        "Perform derivative precomputations and objective function evaluation. Returns f(xk).\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n");
    m.def(
        "initialize_solve",
        [](ElastoDynamics& fem, Params& params) {
            pbat::sim::algorithm::newton::InitializeSolve<ElasticEnergyType>(fem, params);
        },
        nb::arg("fem"),
        nb::arg("params"),
        "Initialize Newton solve (computes initial derivatives and gradient).\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n");
    m.def(
        "iterate",
        [](ElastoDynamics& fem, Params& params) {
            return pbat::sim::algorithm::newton::Iterate<ElasticEnergyType>(fem, params);
        },
        nb::arg("fem"),
        nb::arg("params"),
        "Perform one Newton iteration; returns True if a step was taken.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n"
        "Returns:\n"
        "    bool: True if a step was taken, False otherwise.");
    m.def(
        "solve",
        [](ElastoDynamics& fem, Params& params) {
            return pbat::sim::algorithm::newton::Solve<ElasticEnergyType>(fem, params);
        },
        nb::arg("fem"),
        nb::arg("params"),
        "Run Newton's method to convergence (or until max iterations); returns True on "
        "convergence.\n\n"
        "Args:\n"
        "    fem (FemElastoDynamics): Finite element elasto dynamics problem.\n"
        "    params (Params): Newton solver parameters.\n"
        "Returns:\n"
        "    bool: True if convergence is achieved, False otherwise.");
}

} // namespace pbat::py::sim::algorithm::newton
