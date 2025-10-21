#include "Core.h"

#include <nanobind/eigen/dense.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/physics/Enums.h>
#include <pbat/physics/SaintVenantKirchhoffEnergy.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pbat/sim/algorithm/vbd/Core.h>
#include <pbat/sim/algorithm/vbd/Enums.h>
#include <pbat/sim/dynamics/FemElastoDynamics.h>

namespace pbat::py::sim::algorithm::vbd {

void BindCore(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using IndexType  = Index;
    using pbat::sim::algorithm::vbd::EInitializationStrategy;
    using pbat::sim::algorithm::vbd::FemElastoDynamics;
    using pbat::sim::algorithm::vbd::Params;

    nb::enum_<EInitializationStrategy>(m, "EInitializationStrategy")
        .value("Position", EInitializationStrategy::Position)
        .value("Inertia", EInitializationStrategy::Inertia)
        .value("KineticEnergyMinimum", EInitializationStrategy::KineticEnergyMinimum)
        .value("AdaptiveVbd", EInitializationStrategy::AdaptiveVbd)
        .value("AdaptivePbat", EInitializationStrategy::AdaptivePbat)
        .export_values();

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
            nb::arg("colors"),
            nb::rv_policy::reference_internal,
            "Vertex colors used for coloring the VBD solve.\n\n"
            "Args:\n"
            "    colors (numpy.ndarray): `|# verts|` vertex colors\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_initialization_strategy",
            &Params::WithInitializationStrategy,
            nb::arg("strategy"),
            nb::rv_policy::reference_internal,
            "Initialization strategy for the VBD solver.\n\n"
            "Args:\n"
            "    strategy (pbat.sim.algorithm.vbd.EInitializationStrategy): Initialization "
            "strategy\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_maximum_iterations",
            &Params::WithMaximumIterations,
            nb::arg("n_iters"),
            nb::rv_policy::reference_internal,
            "Maximum number of VBD iterations.\n\n"
            "Args:\n"
            "    n_iters (int): Maximum number of iterations\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_hessian_determinant_zero",
            &Params::WithHessianDeterminantZeroUnder,
            nb::arg("zero"),
            nb::rv_policy::reference_internal,
            "Numerical zero for hessian pseudo-singularity check.\n\n"
            "Args:\n"
            "    zero (float): Numerical zero\n"
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
        .def_rw("GVGp", &Params::GVGp, "`|# verts+1|` prefixes into GVGe")
        .def_rw("GVGe", &Params::GVGe, "`|# of vertex-elems adjacencies|` element indices")
        .def_rw(
            "GVGilocal",
            &Params::GVGilocal,
            "`|# of vertex-elems adjacencies|` local vertex indices")
        .def_rw("colors", &Params::colors, "`|# verts|` vertex colors")
        .def_rw(
            "Pptr",
            &Params::Pptr,
            "`|# partitions+1|` partition pointers, s.t. the range `[Pptr[p], Pptr[p+1])` indexes "
            "into Padj from partition `p`")
        .def_rw("Padj", &Params::Padj, "`|# verts|` partition vertices")
        .def_rw("strategy", &Params::strategy, "BCD optimization initialization strategy")
        .def_rw("detH_zero", &Params::detHZero, "Determinant of Hessian zero threshold")
        .def_rw("n_max_iters", &Params::nMaxIters, "Maximum number of iterations");

    pbat::common::ForTypes<
        pbat::physics::StableNeoHookeanEnergy<3>,
        pbat::physics::SaintVenantKirchhoffEnergy<3>>([&]<class TElasticEnergy>() {
        m.def(
            "initialize_solve",
            [](FemElastoDynamics<TElasticEnergy>& fem, Params const& params) {
                pbat::sim::algorithm::vbd::InitializeSolve<TElasticEnergy>(fem, params);
            },
            nb::arg("fem"),
            nb::arg("params"),
            "Initialize the VBD minimization solve.\n\n"
            "Args:\n"
            "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
            "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters");
        m.def(
            "iterate",
            [](FemElastoDynamics<TElasticEnergy>& fem, Params const& params) {
                pbat::sim::algorithm::vbd::Iterate<TElasticEnergy>(fem, params);
            },
            nb::arg("fem"),
            nb::arg("params"),
            "Perform one VBD minimization iteration.\n\n"
            "Args:\n"
            "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
            "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters");
        m.def(
            "solve",
            [](FemElastoDynamics<TElasticEnergy>& fem, Params const& params) {
                pbat::sim::algorithm::vbd::Solve<TElasticEnergy>(fem, params);
            },
            nb::arg("fem"),
            nb::arg("params"),
            "Solve the VBD minimization up to maximum iterations.\n\n"
            "Args:\n"
            "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
            "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters");
        m.def(
            "integrate",
            [](FemElastoDynamics<TElasticEnergy>& fem, Params const& params) {
                pbat::sim::algorithm::vbd::Integrate<TElasticEnergy>(fem, params);
            },
            nb::arg("fem"),
            nb::arg("params"),
            "Integrate one time step using VBD as non-linear solver.\n\n"
            "Args:\n"
            "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
            "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters");
    });
}

} // namespace pbat::py::sim::algorithm::vbd
