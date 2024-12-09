#include "Hierarchy.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/multigrid/Hierarchy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindHierarchy(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::multigrid::Hierarchy;
    pyb::class_<Hierarchy>(m, "Hierarchy")
        .def(
            pyb::init([](Data& root,
                         std::vector<Eigen::Ref<MatrixX const>> const& X,
                         std::vector<Eigen::Ref<IndexMatrixX const>> const& E,
                         Eigen::Ref<IndexMatrixX const> const& cycle,
                         Eigen::Ref<IndexVectorX const> const& transitionSchedule,
                         Eigen::Ref<IndexVectorX const> const& smoothingSchedule) {
                return Hierarchy(
                    std::move(root),
                    X,
                    E,
                    cycle,
                    transitionSchedule,
                    smoothingSchedule);
            }),
            pyb::arg("root"),
            pyb::arg("X"),
            pyb::arg("E"),
            pyb::arg("cycle")               = IndexMatrixX{},
            pyb::arg("transition_schedule") = IndexVectorX{},
            pyb::arg("smoothing_schedule")  = IndexVectorX{},
            "Computes a geometric multigrid hierarchy from the full space root problem, given an "
            "ordered list of coarse embedding/cage meshes (X,E).\n"
            "Args:\n"
            "root (_pbat.sim.vbd.Data): The root problem, defined on the finest (i.e. "
            "full-resolution) mesh.\n"
            "X (list[np.ndarray]): List of cage mesh vertex positions.\n"
            "E (list[np.ndarray]): List of cage mesh tetrahedral elements.\n"
            "cycle (list[(int, int)] | None): List of level transitions (li,lj), where the root "
            "level is -1, the immediate coarse level is 0, etc. Defaults to None.\n"
            "transition_schedule (list[int] | None): |len(cycle)| list of iterations to spend on "
            "each transition. Defaults to None.\n"
            "smoothing_schedule (list[int] | None): |len(cycle)+1| list of iterations to spend on "
            "each visited level in the cycle. Defaults to None.")
        .def_readwrite("root", &Hierarchy::mRoot)
        .def_readwrite("levels", &Hierarchy::mLevels)
        .def_readwrite("cycle", &Hierarchy::mCycle)
        .def_readwrite("transition_schedule", &Hierarchy::mTransitionSchedule)
        .def_readwrite("smoothing_schedule", &Hierarchy::mSmoothingSchedule)
        .def_readwrite("transitions", &Hierarchy::mTransitions);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat