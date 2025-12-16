#include "Chebyshev.h"

#include <nanobind/eigen/dense.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/physics/Enums.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pbat/sim/algorithm/vbd/Chebyshev.h>
#include <pbat/sim/algorithm/vbd/Enums.h>
#include <pbat/sim/contact/MeshDynamics.h>
#include <pbat/sim/dynamics/FemElastoDynamics.h>

namespace pbat::py::sim::algorithm::vbd {

void BindChebyshev(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using pbat::sim::algorithm::common::FemElastoDynamics;
    using pbat::sim::algorithm::vbd::Params;
    using ChebyshevParams = pbat::sim::algorithm::vbd::ChebyshevParams;

    nb::class_<ChebyshevParams>(m, "ChebyshevParams")
        .def(nb::init<>())
        .def(
            "serialize",
            &ChebyshevParams::Serialize,
            nb::arg("archive"),
            "Serialize this to archive.")
        .def(
            "deserialize",
            &ChebyshevParams::Deserialize,
            nb::arg("archive"),
            "Deserialize this from archive.")
        .def_rw("rho", &ChebyshevParams::rho, "Spectral radius estimate")
        .def_ro("rho2", &ChebyshevParams::rho2, "Square of spectral radius estimate")
        .def_ro("omega", &ChebyshevParams::omega, "Relaxation parameter")
        .def_ro("xkm1", &ChebyshevParams::xkm1, "Previous iterate")
        .def_ro("xkm2", &ChebyshevParams::xkm2, "Second previous iterate")
        .def_ro("k", &ChebyshevParams::k, "Current iteration index");

    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using MeshDynamicsType  = pbat::sim::contact::MeshDynamics<ScalarType, IndexType>;

    m.def(
        "initialize_solve",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType& meshDynamics,
           Params& params,
           ChebyshevParams& cheb) {
            pbat::sim::algorithm::vbd::InitializeSolve<ElasticEnergyType>(
                fem,
                meshDynamics,
                params,
                cheb);
        },
        nb::arg("fem"),
        nb::arg("mesh_dynamics"),
        nb::arg("params"),
        nb::arg("cheb"),
        "Initialize Chebyshev accelerated VBD minimization solve.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    mesh_dynamics (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "    cheb (pbat.sim.algorithm.vbd.ChebyshevParams): The Chebyshev parameters");
    m.def(
        "iterate",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType& meshDynamics,
           Params& params,
           ChebyshevParams& cheb) {
            pbat::sim::algorithm::vbd::Iterate<ElasticEnergyType>(fem, meshDynamics, params, cheb);
        },
        nb::arg("fem"),
        nb::arg("mesh_dynamics"),
        nb::arg("params"),
        nb::arg("cheb"),
        "Perform a single Chebyshev accelerated VBD minimization iteration.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    mesh_dynamics (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "    cheb (pbat.sim.algorithm.vbd.ChebyshevParams): The Chebyshev parameters");
    m.def(
        "solve",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType& meshDynamics,
           Params& params,
           ChebyshevParams& cheb) {
            pbat::sim::algorithm::vbd::Solve<ElasticEnergyType>(fem, meshDynamics, params, cheb);
        },
        nb::arg("fem"),
        nb::arg("mesh_dynamics"),
        nb::arg("params"),
        nb::arg("cheb"),
        "Solve the VBD minimization.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    mesh_dynamics (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "    cheb (pbat.sim.algorithm.vbd.ChebyshevParams): The Chebyshev parameters");
    m.def(
        "integrate",
        [](FemElastoDynamics<ElasticEnergyType>& fem,
           MeshDynamicsType& meshDynamics,
           Params& params,
           ChebyshevParams& cheb) {
            pbat::sim::algorithm::vbd::Integrate<ElasticEnergyType>(
                fem,
                meshDynamics,
                params,
                cheb);
        },
        nb::arg("fem"),
        nb::arg("mesh_dynamics"),
        nb::arg("params"),
        nb::arg("cheb"),
        "Integrate one time step using Chebyshev accelerated VBD minimization.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    mesh_dynamics (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n"
        "    cheb (pbat.sim.algorithm.vbd.ChebyshevParams): The Chebyshev parameters");
}

} // namespace pbat::py::sim::algorithm::vbd
