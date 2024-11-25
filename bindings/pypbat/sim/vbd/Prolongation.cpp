#include "Prolongation.h"

#include <pbat/sim/vbd/Hierarchy.h>
#include <pbat/sim/vbd/Prolongation.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindProlongation(pybind11::module& m)
{
    namespace pyb = pybind11;

    using pbat::sim::vbd::Hierarchy;
    using pbat::sim::vbd::Prolongation;

    pyb::class_<Prolongation>(m, "Prolongation")
        .def(pyb::init<>())
        .def(
            "from",
            &Prolongation::From,
            pyb::arg("lc"),
            "Associates this Prolongation operator with coarse level lc.\n"
            "Args:\n"
            "lc (int): Coarse level index (-1 is root, i.e. finest)")
        .def(
            "to",
            &Prolongation::To,
            pyb::arg("lf"),
            "Associates this Prolongation operator with fine level lf.\n"
            "Args:\n"
            "lf (int): Fine level index (-1 is root, i.e. finest)")
        .def(
            "with_coarse_shape_functions",
            &Prolongation::WithCoarseShapeFunctions,
            pyb::arg("ec"),
            pyb::arg("Nc"),
            "Sets the coarse level shape functions Nc and their associated elements ec used to "
            "interpolate the coarse level lc's solution to the fine level lf.\n"
            "Args:\n"
            "ec (np.ndarray): |#verts at fine level| array of coarse cage elements containing fine "
            "level vertices\n"
            "Nc (np.ndarray): 4x|#verts at fine level| array of coarse cage shape functions at "
            "fine level vertices")
        .def("construct", &Prolongation::Construct, pyb::arg("validate") = true)
        .def(
            "apply",
            &Prolongation::Apply,
            pyb::arg("hierarchy"),
            "Prolongs level lc to level lf of hierarchy.")
        .def_readwrite("ec", &Prolongation::ec)
        .def_readwrite("Nc", &Prolongation::Nc)
        .def_readwrite("lc", &Prolongation::lc)
        .def_readwrite("lf", &Prolongation::lf);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat