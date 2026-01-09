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
    using pbat::sim::algorithm::vbd::EHomogenizationStrategy;
    using pbat::sim::algorithm::vbd::EInitializationStrategy;
    using pbat::sim::algorithm::vbd::Params;

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
            pbat::sim::algorithm::vbd::VertexColors(E, nNodes, eOrdering, eSelection, colors);
            return colors;
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
        "    numpy.ndarray: `|# verts| x 1` Vertex colors");

    nb::enum_<EHomogenizationStrategy>(m, "EHomogenizationStrategy")
        .value("Off", EHomogenizationStrategy::None, "No homogenization")
        .value(
            "Sensitivity",
            EHomogenizationStrategy::Sensitivity,
            "Homogenize using sensitivity histogram")
        .value(
            "Conditioning",
            EHomogenizationStrategy::Conditioning,
            "Homogenize using conditioning histogram");

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
            "Maximum number of VBD iterations.\n\n"
            "Args:\n"
            "    n_iters (int): Maximum number of iterations\n"
            "Returns:\n"
            "    self (pbat.sim.algorithm.vbd.Params): Reference to this")
        .def(
            "with_homogenization",
            &Params::WithHomogenization,
            nb::arg("strategy"),
            nb::rv_policy::reference_internal,
            "Homogenization strategy.\n\n"
            "Args:\n"
            "    strategy (pbat.sim.algorithm.vbd.EHomogenizationStrategy): Homogenization "
            "strategy\n"
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
        .def("serialize", &Params::Serialize, nb::arg("archive"), "Serialize this to archive.")
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
        .def_rw("colors", &Params::colors, "`|# verts|` vertex colors")
        .def_rw(
            "Pptr",
            &Params::Pptr,
            "`|# partitions+1|` partition pointers, s.t. the range `[Pptr[p], Pptr[p+1])` indexes "
            "into Padj from partition `p`")
        .def_rw("Padj", &Params::Padj, "`|# verts|` partition vertices")
        .def_rw("betaR", &Params::betaR, "Rayleigh damping coefficient")
        .def_rw("n_max_iters", &Params::nMaxIters, "Maximum number of iterations")
        .def_rw("detH_zero", &Params::detHZero, "Determinant of Hessian zero threshold")
        .def_ro("smin", &Params::smin, "`|# nodes|` minimum sensitivities")
        .def_ro("k", &Params::k, "Current iteration");

    using ElasticEnergyType = pbat::physics::StableNeoHookeanEnergy<3>;
    using MeshDynamicsType  = pbat::sim::contact::MeshDynamics<ScalarType, IndexType>;

    m.def(
        "iterate",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            pbat::sim::algorithm::vbd::Iterate<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Perform one VBD minimization iteration.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n");
    m.def(
        "initialize_solve",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            pbat::sim::algorithm::vbd::InitializeSolve<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Initialize the VBD minimization solve.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters\n");
    m.def(
        "solve",
        [](FemElastoDynamics<ElasticEnergyType>& fem, MeshDynamicsType& contact, Params& params) {
            pbat::sim::algorithm::vbd::Solve<ElasticEnergyType>(fem, contact, params);
        },
        nb::arg("fem"),
        nb::arg("contact"),
        nb::arg("params"),
        "Solve the VBD minimization.\n\n"
        "Args:\n"
        "    fem (pbat.sim.dynamics.FemElastoDynamics): The FEM elasto-dynamics system\n"
        "    contact (pbat.sim.contact.MeshDynamics): The mesh contact dynamics system\n"
        "    params (pbat.sim.algorithm.vbd.Params): The VBD parameters");
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
