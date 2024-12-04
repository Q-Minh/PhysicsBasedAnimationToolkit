#include "Restriction.h"

#include <pbat/sim/vbd/Hierarchy.h>
#include <pbat/sim/vbd/Restriction.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindRestriction(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Hierarchy;
    using pbat::sim::vbd::Restriction;

    pyb::class_<Restriction>(m, "Restriction")
        .def(pyb::init<>())
        .def(
            "from_level",
            &Restriction::From,
            pyb::arg("lf"),
            "Associates this Restriction operator with fine level lf.\n"
            "Args:\n"
            "lf (int): Fine level index (-1 is root, i.e. finest)")
        .def(
            "to_level",
            &Restriction::To,
            pyb::arg("lc"),
            "Associates this Restriction operator with coarse level lc.\n"
            "Args:\n"
            "lc (int): Coarse level index (-1 is root, i.e. finest)")
        .def(
            "with_fine_shape_functions",
            &Restriction::WithFineShapeFunctions,
            pyb::arg("efg"),
            pyb::arg("Nfg"),
            "Sets the fine level shape functions Nfg and their associated elements efg used to "
            "interpolate the fine level lc's solution at coarse quadrature points g.\n"
            "Args:\n"
            "efg (np.ndarray): |#quad.pts.| array of fine cage elements associated with quadrature "
            "points\n"
            "Nfg (np.ndarray): 4x|#quad.pts.| array of fine cage shape functions at coarse "
            "quadrature points")
        .def(
            "iterate",
            &Restriction::Iterate,
            pyb::arg("iters"),
            "Set the number of descent iterations to compute restriction")
        .def("construct", &Restriction::Construct, pyb::arg("validate") = true)
        .def(
            "apply",
            &Restriction::Apply,
            pyb::arg("hierarchy"),
            "Restricts level lf to level lc of hierarchy.")
        .def_readwrite("efg", &Restriction::efg)
        .def_readwrite("Nfg", &Restriction::Nfg)
        .def_readwrite("xfg", &Restriction::xfg)
        .def_readwrite("iters", &Restriction::iterations)
        .def_readwrite("lc", &Restriction::lc)
        .def_readwrite("lf", &Restriction::lf);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat
