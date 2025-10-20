#include "Broyden.h"

#include <nanobind/eigen/dense.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/physics/Enums.h>
#include <pbat/physics/SaintVenantKirchhoffEnergy.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pbat/sim/algorithm/vbd/Broyden.h>
#include <pbat/sim/algorithm/vbd/Enums.h>
#include <pbat/sim/dynamics/FemElastoDynamics.h>

namespace pbat::py::sim::algorithm::vbd {

void BindBroyden(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using pbat::sim::algorithm::vbd::FemElastoDynamics;
    using pbat::sim::algorithm::vbd::Params;
    using BroydenParams = pbat::sim::algorithm::vbd::BroydenParams;
    using pbat::sim::algorithm::vbd::EBroydenJacobianEstimate;
    using pbat::sim::algorithm::vbd::EBroydenLeastSquaresSolver;

    nb::enum_<EBroydenLeastSquaresSolver>(m, "EBroydenLeastSquaresSolver")
        .value("QR", EBroydenLeastSquaresSolver::QR)
        .value("COD", EBroydenLeastSquaresSolver::COD)
        .value("LSCG", EBroydenLeastSquaresSolver::LSCG)
        .value("OneStepSteepestDescent", EBroydenLeastSquaresSolver::OneStepSteepestDescent)
        .export_values();

    nb::enum_<EBroydenJacobianEstimate>(m, "EBroydenJacobianEstimate")
        .value("Identity", EBroydenJacobianEstimate::Identity)
        .value("ScaledIdentity", EBroydenJacobianEstimate::ScaledIdentity)
        .value(
            "QuasiCauchyRelationDiagonalUpdating",
            EBroydenJacobianEstimate::QuasiCauchyRelationDiagonalUpdating)
        .value("UsdDiagonal", EBroydenJacobianEstimate::UsdDiagonal)
        .value("DiagonalCauchySchwarz", EBroydenJacobianEstimate::DiagonalCauchySchwarz)
        .export_values();

    nb::class_<BroydenParams>(m, "BroydenParams")
        .def(nb::init<>())
        .def_rw("m", &BroydenParams::m, "Window size")
        .def_rw("eps_l2_solve", &BroydenParams::epsL2Solve, "L2 solve tolerance")
        .def_rw(
            "max_l2_solver_iters",
            &BroydenParams::maxL2SolverIters,
            "Maximum L2 solver iterations")
        .def_rw("l2_solver", &BroydenParams::eL2Solver, "L2 solver type")
        .def_rw(
            "jacobian_estimate",
            &BroydenParams::eJacobianEstimate,
            "Jacobian estimate strategy")
        .def_rw(
            "broyden_beta_F",
            &BroydenParams::betaF,
            "Rank estimate for Fk in diagonal Cauchy-Schwarz updating")
        .def_rw(
            "broyden_beta_B",
            &BroydenParams::betaB,
            "Rank estimate for Bk in diagonal Cauchy-Schwarz updating")
        .def_ro("k", &BroydenParams::k, "Current iteration")
        .def_ro("Fk", &BroydenParams::Fk, "`|# dofs| x m` residual differences")
        .def_ro("Xk", &BroydenParams::Xk, "`|# dofs| x m` past step differences")
        .def_ro("xkm1", &BroydenParams::xkm1, "`|# dofs| x 1` previous step")
        .def_ro("fk", &BroydenParams::fk, "`|# dofs| x 1` current residual")
        .def_ro("fkm1", &BroydenParams::fkm1, "`|# dofs| x 1` past residual")
        .def_ro("gammak", &BroydenParams::gammak, "`m x 1` subspace residual")
        .def_ro(
            "FkRowNorm2",
            &BroydenParams::FkRowNorm2,
            "`|# dofs| x m` Cauchy-Schwarz squared norms on rows of Fk")
        .def_ro("Gkm", &BroydenParams::Gkm, "`|# dofs| x m` diag(G_{k-m})")
        .def_ro("Sigma", &BroydenParams::Sigma, "`m x 1` scaled identity coefficients window")
        .def_ro(
            "sqrtBetaB",
            &BroydenParams::sqrtBetaB,
            "Cached sqrt(betaB) for diagonal Cauchy-Schwarz updating")
        .def_ro(
            "Fknorm2",
            &BroydenParams::Fknorm2,
            "Cached ||F_k||_F^2 for diagonal Cauchy-Schwarz updating")
        .def_ro(
            "Bknorm2",
            &BroydenParams::Bknorm2,
            "Cached ||B_k||_F^2 for diagonal Cauchy-Schwarz updating")
        .def_ro("gradL2", &BroydenParams::gradL2, "`m x 1` least-squares gradient")
        .def_ro("FkgradL2", &BroydenParams::FkgradL2, "`|# dofs| x 1` Fk * gradL2");

    pbat::common::ForTypes<
        pbat::physics::StableNeoHookeanEnergy<3>,
        pbat::physics::SaintVenantKirchhoffEnergy<3>>([&]<class TElasticEnergy>() {
        m.def(
            "initialize_solve",
            [](FemElastoDynamics<TElasticEnergy>& fem,
               Params const& params,
               BroydenParams& broyden) {
                pbat::sim::algorithm::vbd::InitializeSolve<TElasticEnergy>(fem, params, broyden);
            },
            nb::arg("fem"),
            nb::arg("params"),
            nb::arg("broyden"),
            "Initialize the Broyden solver for VBD.\n\n"
            "Args:\n"
            "    fem (FemElastoDynamics): The FEM elastodynamics simulator\n"
            "    params (Params): The VBD parameters\n"
            "    broyden (BroydenParams): The Broyden parameters\n");
        m.def(
            "iterate",
            [](FemElastoDynamics<TElasticEnergy>& fem,
               Params const& params,
               BroydenParams& broyden) {
                pbat::sim::algorithm::vbd::Iterate<TElasticEnergy>(fem, params, broyden);
            },
            nb::arg("fem"),
            nb::arg("params"),
            nb::arg("broyden"),
            "Perform one Broyden-accelerated VBD minimization iteration.\n\n"
            "Args:\n"
            "    fem (FemElastoDynamics): The FEM elastodynamics simulator\n"
            "    params (Params): The VBD parameters\n"
            "    broyden (BroydenParams): The Broyden parameters\n");
        m.def(
            "solve",
            [](FemElastoDynamics<TElasticEnergy>& fem,
               Params const& params,
               BroydenParams& broyden) {
                pbat::sim::algorithm::vbd::Solve<TElasticEnergy>(fem, params, broyden);
            },
            nb::arg("fem"),
            nb::arg("params"),
            nb::arg("broyden"),
            "Solve the Broyden accelerated VBD minimization problem.\n\n"
            "Args:\n"
            "    fem (FemElastoDynamics): The FEM elastodynamics simulator\n"
            "    params (Params): The VBD parameters\n"
            "    broyden (BroydenParams): The Broyden parameters\n");
        m.def(
            "integrate",
            [](FemElastoDynamics<TElasticEnergy>& fem,
               Params const& params,
               BroydenParams& broyden) {
                pbat::sim::algorithm::vbd::Integrate<TElasticEnergy>(fem, params, broyden);
            },
            nb::arg("fem"),
            nb::arg("params"),
            nb::arg("broyden"),
            "Integrate one time step using VBD as non-linear solver.\n\n"
            "Args:\n"
            "    fem (FemElastoDynamics): The FEM elastodynamics simulator\n"
            "    params (Params): The VBD parameters\n"
            "    broyden (BroydenParams): The Broyden parameters\n");
    });
}

} // namespace pbat::py::sim::algorithm::vbd
