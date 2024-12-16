#include "Level.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Mesh.h>
#include <pbat/sim/vbd/multigrid/Level.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindLevel(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::py::fem::Mesh;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::VolumeMesh;
    using pbat::sim::vbd::multigrid::Level;
    pyb::class_<Level>(m, "Level")
        .def(
            pyb::init([](Data const& root, Mesh const& cage) {
                VolumeMesh const* cageRaw = cage.Raw<VolumeMesh>();
                if (cageRaw == nullptr)
                    throw std::invalid_argument(
                        "Requested underlying MeshType that this Mesh does not hold.");
                return Level(root, *cageRaw);
            }),
            pyb::arg("data"),
            pyb::arg("cage"),
            "Computes a level of a geometric multigrid hierarchy from the full space root problem, "
            "given a coarse embedding/cage mesh.\n"
            "Args:\n"
            "root (_pbat.sim.vbd.Data): The root problem, defined on the finest (i.e. "
            "full-resolution) mesh.\n"
            "cage (_pbat.fem.Mesh): Cage mesh.\n")
        .def("smooth", &Level::Smooth, pyb::arg("dt"), pyb::arg("iters"), pyb::arg("data"))
        .def("prolong", &Level::Prolong, pyb::arg("data"))
        .def_property(
            "X",
            [](Level const& l) { return l.mesh.X; },
            [](Level& l, Eigen::Ref<MatrixX const> const& X) { l.mesh.X = X; },
            "3x|#coarse nodes| cage nodes")
        .def_property(
            "E",
            [](Level const& l) { return l.mesh.E; },
            [](Level& l, Eigen::Ref<IndexMatrixX const> const& E) { l.mesh.E = E; },
            "4x|#coarse elements| cage elements")
        .def_readwrite("u", &Level::u, "3x|#coarse nodes| displacement coefficients")
        .def_readwrite("colors", &Level::colors, "Coarse vertex coloring")
        .def_readwrite("Pindptr", &Level::Pptr, "Parallel vertex partition pointers")
        .def_readwrite("Pindices", &Level::Padj, "Parallel vertex partition vertex indices")
        .def_readwrite(
            "ecVE",
            &Level::ecVE,
            "4x|#fine elems| coarse elements containing 4 vertices of fine elements")
        .def_readwrite(
            "NecVE",
            &Level::NecVE,
            "4x|4*#fine elems| coarse element shape functions at 4 vertices of fine elements")
        .def_readwrite(
            "ilocalE",
            &Level::ilocalE,
            "4x|#fine elems| coarse vertex local index w.r.t. coarse elements containing 4 "
            "vertices of fine elements")
        .def_readwrite(
            "GEindptr",
            &Level::GEptr,
            "Coarse vertex -> fine element adjacency graph pointers")
        .def_readwrite(
            "GEindices",
            &Level::GEadj,
            "Coarse vertex -> fine element adjacency graph indices")
        .def_readwrite(
            "ecK",
            &Level::ecK,
            "|#fine vertices| coarse elements containing fine vertices")
        .def_readwrite(
            "NecK",
            &Level::NecK,
            "4x|#fine vertices| coarse element shape functions at fine vertices")
        .def_readwrite(
            "GKindptr",
            &Level::GKptr,
            "Coarse vertex -> fine vertex adjacency graph pointers")
        .def_readwrite(
            "GKindices",
            &Level::GKadj,
            "Coarse vertex -> fine vertex adjacency graph indices")
        .def_readwrite(
            "GKilocal",
            &Level::GKilocal,
            "Coarse vertex -> fine vertex adjacency graph edge weights, i.e. local coarse vertex "
            "indices in embedding coarse elements which contain fine vertices")
        .def_readwrite(
            "is_dirichlet_vertex",
            &Level::bIsDirichletVertex,
            "Boolean mask identifying Dirichlet constrained vertices")
        .def_readwrite("hyper_reduction", &Level::HR, "Hyper reduction scheme at this level");
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat