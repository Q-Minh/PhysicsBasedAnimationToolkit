#include "Chebyshev.h"

#include <nanobind/eigen/dense.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/physics/Enums.h>
#include <pbat/physics/SaintVenantKirchhoffEnergy.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pbat/sim/algorithm/vbd/Chebyshev.h>
#include <pbat/sim/algorithm/vbd/Enums.h>
#include <pbat/sim/dynamics/FemElastoDynamics.h>

namespace pbat::py::sim::algorithm::vbd {

void BindChebyshev(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using pbat::sim::algorithm::vbd::FemElastoDynamics;
    using pbat::sim::algorithm::vbd::Params;
    using ChebyshevParams = pbat::sim::algorithm::vbd::ChebyshevParams;

    nb::class_<ChebyshevParams>(m, "ChebyshevParams")
        .def(nb::init<>())
        .def_rw("rho", &ChebyshevParams::rho, "Spectral radius estimate")
        .def_rw("k", &ChebyshevParams::k, "Iteration")
        .def_ro("rho2", &ChebyshevParams::rho2, "Square of spectral radius estimate")
        .def_ro("omega", &ChebyshevParams::omega, "Relaxation parameter")
        .def_ro("xkm1", &ChebyshevParams::xkm1, "Previous iterate")
        .def_ro("xkm2", &ChebyshevParams::xkm2, "Second previous iterate");

    pbat::common::ForTypes<
        pbat::physics::StableNeoHookeanEnergy<3>,
        pbat::physics::SaintVenantKirchhoffEnergy<3>>([&]<class TElasticEnergy>() {
        m.def(
            "initialize_solve",
            [](FemElastoDynamics<TElasticEnergy>& fem,
               Params const& params,
               ChebyshevParams& cheb) {
                pbat::sim::algorithm::vbd::InitializeSolve<TElasticEnergy>(fem, params, cheb);
            },
            nb::arg("fem"),
            nb::arg("params"),
            nb::arg("cheb"),
            "Initialize Chebyshev accelerated VBD minimization solve.\n\n"
            "Args:\n"
            "    fem (FemElastoDynamics): The FEM elasto-dynamics system\n"
            "    params (Params): The VBD parameters\n"
            "    cheb (ChebyshevParams): The Chebyshev parameters");
        m.def(
            "iterate",
            [](FemElastoDynamics<TElasticEnergy>& fem,
               Params const& params,
               ChebyshevParams& cheb) {
                pbat::sim::algorithm::vbd::Iterate<TElasticEnergy>(fem, params, cheb);
            },
            nb::arg("fem"),
            nb::arg("params"),
            nb::arg("cheb"),
            "Perform a single Chebyshev accelerated VBD minimization iteration.\n\n"
            "Args:\n"
            "    fem (FemElastoDynamics): The FEM elasto-dynamics system\n"
            "    params (Params): The VBD parameters\n"
            "    cheb (ChebyshevParams): The Chebyshev parameters");
        m.def(
            "solve",
            [](FemElastoDynamics<TElasticEnergy>& fem,
               Params const& params,
               ChebyshevParams& cheb) {
                pbat::sim::algorithm::vbd::Solve<TElasticEnergy>(fem, params, cheb);
            },
            nb::arg("fem"),
            nb::arg("params"),
            nb::arg("cheb"),
            "Solve the VBD minimization up to maximum iterations.\n\n"
            "Args:\n"
            "    fem (FemElastoDynamics): The FEM elasto-dynamics system\n"
            "    params (Params): The VBD parameters\n"
            "    cheb (ChebyshevParams): The Chebyshev parameters");
        m.def(
            "integrate",
            [](FemElastoDynamics<TElasticEnergy>& fem,
               Params const& params,
               ChebyshevParams& cheb) {
                pbat::sim::algorithm::vbd::Integrate<TElasticEnergy>(fem, params, cheb);
            },
            nb::arg("fem"),
            nb::arg("params"),
            nb::arg("cheb"),
            "Integrate one time step using Chebyshev accelerated VBD minimization.\n\n"
            "Args:\n"
            "    fem (FemElastoDynamics): The FEM elasto-dynamics system\n"
            "    params (Params): The VBD parameters\n"
            "    cheb (ChebyshevParams): The Chebyshev parameters");
    });
}

} // namespace pbat::py::sim::algorithm::vbd
