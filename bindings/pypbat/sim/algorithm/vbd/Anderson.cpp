#include "Anderson.h"

#include <nanobind/eigen/dense.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/physics/Enums.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pbat/sim/algorithm/vbd/Anderson.h>
#include <pbat/sim/algorithm/vbd/Enums.h>
#include <pbat/sim/contact/MeshDynamics.h>
#include <pbat/sim/dynamics/FemElastoDynamics.h>

namespace pbat::py::sim::algorithm::vbd {

void BindAnderson(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using pbat::sim::algorithm::common::FemElastoDynamics;
    using pbat::sim::algorithm::vbd::Params;
    using AndersonParams = pbat::sim::algorithm::vbd::AndersonParams;
    nb::class_<AndersonParams>(m, "AndersonParams")
        .def(nb::init<>())
        .def(
            "serialize",
            &AndersonParams::Serialize,
            nb::arg("archive"),
            "Serialize this to archive.")
        .def(
            "deserialize",
            &AndersonParams::Deserialize,
            nb::arg("archive"),
            "Deserialize this from archive.")
        .def_rw("m", &AndersonParams::m, "Window size")
        .def_rw("beta", &AndersonParams::beta, "Mixing parameter")
        .def_rw("cod_numerical_zero", &AndersonParams::codNumericalZero, "Numerical zero threshold")
        .def_ro("Fk", &AndersonParams::Fk, "`|# dofs| x m` residual differences")
        .def_ro("Xk", &AndersonParams::Xk, "`|# dofs| x m` past step differences")
        .def_ro("xkm1", &AndersonParams::xkm1, "`|# dofs| x 1` previous step")
        .def_ro("fk", &AndersonParams::fk, "`|# dofs| x 1` current residual")
        .def_ro("fkm1", &AndersonParams::fkm1, "`|# dofs| x 1` past residual")
        .def_ro("gammak", &AndersonParams::gammak, "`m x 1` subspace residual")
        .def_ro("k", &AndersonParams::k, "Current iteration index");

    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using MeshDynamicsType  = pbat::sim::contact::MeshDynamics<ScalarType, IndexType>;

    m.def(
        "initialize_solve",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType& meshDynamics,
           Params& params,
           AndersonParams& anderson) {
            pbat::sim::algorithm::vbd::InitializeSolve<ElasticEnergyType>(
                fem,
                meshDynamics,
                params,
                anderson);
        },
        nb::arg("fem"),
        nb::arg("mesh_dynamics"),
        nb::arg("params"),
        nb::arg("anderson"),
        "Initialize Anderson accelerated VBD minimization solve.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    mesh_dynamics (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "    anderson (pbat.sim.algorithm.vbd.AndersonParams): The Anderson acceleration "
        "parameters");
    m.def(
        "iterate",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType& meshDynamics,
           Params& params,
           AndersonParams& anderson) {
            pbat::sim::algorithm::vbd::Iterate<ElasticEnergyType>(
                fem,
                meshDynamics,
                params,
                anderson);
        },
        nb::arg("fem"),
        nb::arg("mesh_dynamics"),
        nb::arg("params"),
        nb::arg("anderson"),
        "Perform a single Anderson accelerated VBD minimization iteration.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    mesh_dynamics (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "    anderson (pbat.sim.algorithm.vbd.AndersonParams): The Anderson acceleration "
        "parameters");
    m.def(
        "solve",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType& meshDynamics,
           Params& params,
           AndersonParams& anderson) {
            pbat::sim::algorithm::vbd::Solve<ElasticEnergyType>(
                fem,
                meshDynamics,
                params,
                anderson);
        },
        nb::arg("fem"),
        nb::arg("mesh_dynamics"),
        nb::arg("params"),
        nb::arg("anderson"),
        "Solve the Anderson accelerated VBD minimization problem.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    mesh_dynamics (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "    anderson (pbat.sim.algorithm.vbd.AndersonParams): The Anderson acceleration "
        "parameters");
    m.def(
        "integrate",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType& meshDynamics,
           Params& params,
           AndersonParams& anderson) {
            pbat::sim::algorithm::vbd::Integrate<ElasticEnergyType>(
                fem,
                meshDynamics,
                params,
                anderson);
        },
        nb::arg("fem"),
        nb::arg("mesh_dynamics"),
        nb::arg("params"),
        nb::arg("anderson"),
        "Integrate one time step using Anderson accelerated VBD minimization.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    mesh_dynamics (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "    anderson (pbat.sim.algorithm.vbd.AndersonParams): The Anderson acceleration "
        "parameters");
}

} // namespace pbat::py::sim::algorithm::vbd
