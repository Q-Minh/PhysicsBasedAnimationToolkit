#include "Hierarchy.h"

#include "pypbat/fem/Mesh.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <pbat/sim/vbd/Mesh.h>
#include <pbat/sim/vbd/multigrid/Hierarchy.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindHierarchy(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::py::fem::Mesh;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::VolumeMesh;
    using pbat::sim::vbd::multigrid::Hierarchy;
    nb::class_<Hierarchy>(m, "Hierarchy")
        .def(
            "__init__",
            [](Hierarchy* self,
               Data& root,
               std::vector<Mesh> const& cagesIn,
               Eigen::Ref<IndexVectorX const> const& cycle,
               Eigen::Ref<IndexVectorX const> const& siters) {
                std::vector<VolumeMesh> cages{};
                cages.reserve(cagesIn.size());
                for (auto const& cageIn : cagesIn)
                {
                    VolumeMesh const* cageRaw = cageIn.Raw<VolumeMesh>();
                    if (cageRaw == nullptr)
                        throw std::invalid_argument(
                            "Requested underlying MeshType that this Mesh does not hold.");
                    cages.push_back(*cageRaw);
                }
                new (self) Hierarchy(std::move(root), cages, cycle, siters);
            },
            nb::arg("root"),
            nb::arg("cages"),
            nb::arg("cycle")  = IndexVectorX{},
            nb::arg("siters") = IndexVectorX{},
            "Computes a geometric multigrid hierarchy from the full space root problem, given an "
            "ordered list of coarse embedding/cage meshes (X,E).\n"
            "Args:\n"
            "root (_pbat.sim.vbd.Data): The root problem, defined on the finest (i.e. "
            "full-resolution) mesh.\n"
            "cages (list[_pbat.fem.Mesh]): List of cage meshes.\n"
            "cycle (list[int] | None): List of level transitions (l[i],l[i+1]), where the root "
            "level is -1, the immediate coarse level is 0, etc. Defaults to None.\n"
            "siters (list[int] | None): |len(cycle)| list of iterations to spend on "
            "each visited level in the cycle. Defaults to None.\n")
        .def_rw("data", &Hierarchy::data)
        .def_rw(
            "levels",
            &Hierarchy::levels,
            "Ordered coarse levels with the only requirement that levels[l] embeds data")
        .def_rw(
            "cycle",
            &Hierarchy::cycle,
            "|#level visits| ordered array of levels to visit during the solve. Level -1 is the "
            "root, 0 the first coarse level, etc.")
        .def_rw(
            "siters",
            &Hierarchy::siters,
            "|#cages+1| max smoother iterations at each level, starting from the root");
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat