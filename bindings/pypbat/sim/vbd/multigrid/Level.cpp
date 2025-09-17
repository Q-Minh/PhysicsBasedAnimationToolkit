#include "Level.h"

#include "pypbat/fem/Mesh.h"

#include <nanobind/eigen/dense.h>
#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Mesh.h>
#include <pbat/sim/vbd/multigrid/Level.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindLevel(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::py::fem::Mesh;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::VolumeMesh;
    using pbat::sim::vbd::multigrid::Level;
    nb::class_<Level>(m, "Level")
        .def(
            "__init__",
            [](Level* self, Data const& root, Mesh const& cage) {
                VolumeMesh const* cageRaw = cage.Raw<VolumeMesh>();
                if (cageRaw == nullptr)
                    throw std::invalid_argument(
                        "Requested underlying MeshType that this Mesh does not hold.");
                new (self) Level(root, *cageRaw);
            },
            nb::arg("data"),
            nb::arg("cage"),
            "Computes a level of a geometric multigrid hierarchy from the full space root problem, "
            "given a coarse embedding/cage mesh.\n"
            "Args:\n"
            "root (_pbat.sim.vbd.Data): The root problem, defined on the finest (i.e. "
            "full-resolution) mesh.\n"
            "cage (_pbat.fem.Mesh): Cage mesh.\n")
        .def("smooth", &Level::Smooth, nb::arg("dt"), nb::arg("iters"), nb::arg("data"))
        .def("prolong", &Level::Prolong, nb::arg("data"))
        .def_prop_rw(
            "X",
            [](Level const& l) { return l.mesh.X; },
            [](Level& l, Eigen::Ref<MatrixX const> const& X) { l.mesh.X = X; },
            "3x|#coarse nodes| cage nodes")
        .def_prop_rw(
            "E",
            [](Level const& l) { return l.mesh.E; },
            [](Level& l, Eigen::Ref<IndexMatrixX const> const& E) { l.mesh.E = E; },
            "4x|#coarse elements| cage elements")
        .def_rw("u", &Level::u, "3x|#coarse nodes| displacement coefficients")
        .def_rw("colors", &Level::colors, "Coarse vertex coloring")
        .def_rw("Pindptr", &Level::Pptr, "Parallel vertex partition pointers")
        .def_rw("Pindices", &Level::Padj, "Parallel vertex partition vertex indices")
        .def_rw(
            "ecVE",
            &Level::ecVE,
            "4x|#fine elems| coarse elements containing 4 vertices of fine elements")
        .def_rw(
            "NecVE",
            &Level::NecVE,
            "4x|4*#fine elems| coarse element shape functions at 4 vertices of fine elements")
        .def_rw(
            "ilocalE",
            &Level::ilocalE,
            "4x|#fine elems| coarse vertex local index w.r.t. coarse elements containing 4 "
            "vertices of fine elements")
        .def_rw("GEindptr", &Level::GEptr, "Coarse vertex -> fine element adjacency graph pointers")
        .def_rw("GEindices", &Level::GEadj, "Coarse vertex -> fine element adjacency graph indices")
        .def_rw("ecK", &Level::ecK, "|#fine vertices| coarse elements containing fine vertices")
        .def_rw(
            "NecK",
            &Level::NecK,
            "4x|#fine vertices| coarse element shape functions at fine vertices")
        .def_rw("GKindptr", &Level::GKptr, "Coarse vertex -> fine vertex adjacency graph pointers")
        .def_rw("GKindices", &Level::GKadj, "Coarse vertex -> fine vertex adjacency graph indices")
        .def_rw(
            "GKilocal",
            &Level::GKilocal,
            "Coarse vertex -> fine vertex adjacency graph edge weights, i.e. local coarse vertex "
            "indices in embedding coarse elements which contain fine vertices")
        .def_rw(
            "is_dirichlet_vertex",
            &Level::bIsDirichletVertex,
            "Boolean mask identifying Dirichlet constrained vertices");
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat