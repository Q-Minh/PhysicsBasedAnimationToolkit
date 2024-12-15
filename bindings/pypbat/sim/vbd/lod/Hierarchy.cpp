#include "Hierarchy.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Mesh.h>
#include <pbat/sim/vbd/lod/Hierarchy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindHierarchy(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::py::fem::Mesh;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::VolumeMesh;
    using pbat::sim::vbd::lod::CageQuadratureParameters;
    using pbat::sim::vbd::lod::Hierarchy;
    pyb::class_<Hierarchy>(m, "Hierarchy")
        .def(
            pyb::init([](Data& root,
                         std::vector<Mesh> const& cagesIn,
                         Eigen::Ref<IndexVectorX const> const& cycle,
                         Eigen::Ref<IndexVectorX const> const& smoothingSchedule,
                         Eigen::Ref<IndexVectorX const> const& transitionSchedule,
                         std::vector<CageQuadratureParameters> const& cageQuadParams) {
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
                return Hierarchy(
                    std::move(root),
                    cages,
                    cycle,
                    smoothingSchedule,
                    transitionSchedule,
                    cageQuadParams);
            }),
            pyb::arg("root"),
            pyb::arg("cages"),
            pyb::arg("cycle")               = IndexVectorX{},
            pyb::arg("smoothing_schedule")  = IndexVectorX{},
            pyb::arg("transition_schedule") = IndexVectorX{},
            pyb::arg("cage_quad_params")    = std::vector<CageQuadratureParameters>{},
            "Computes a geometric lod hierarchy from the full space root problem, given an "
            "ordered list of coarse embedding/cage meshes (X,E).\n"
            "Args:\n"
            "root (_pbat.sim.vbd.Data): The root problem, defined on the finest (i.e. "
            "full-resolution) mesh.\n"
            "cages (list[_pbat.fem.Mesh]): List of cage meshes.\n"
            "cycle (list[int] | None): List of level transitions (l[i],l[i+1]), where the root "
            "level is -1, the immediate coarse level is 0, etc. Defaults to None.\n"
            "smoothing_schedule (list[int] | None): |len(cycle)| list of iterations to spend on "
            "each visited level in the cycle. Defaults to None.\n"
            "transition_schedule (list[int] | None): |len(cycle)-1| list of iterations to spend on "
            "each transition. Defaults to None.\n"
            "cage_quad_params (list[_pbat.sim.vbd.lod.CageQuadratureParameters] | None): "
            "|len(X)| list of parameters to create cage quadratures at each level. Defaults to "
            "None.")
        .def_readwrite("root", &Hierarchy::mRoot)
        .def_readwrite("levels", &Hierarchy::mLevels)
        .def_readwrite("cycle", &Hierarchy::mCycle)
        .def_readwrite("smoothing_schedule", &Hierarchy::mSmoothingSchedule)
        .def_readwrite("transition_schedule", &Hierarchy::mTransitionSchedule)
        .def_readwrite("transitions", &Hierarchy::mTransitions);
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat