#include "Restriction.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/multigrid/Level.h>
#include <pbat/sim/vbd/multigrid/Mesh.h>
#include <pbat/sim/vbd/multigrid/Quadrature.h>
#include <pbat/sim/vbd/multigrid/Restriction.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindRestriction(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::py::fem::Mesh;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::multigrid::CageQuadrature;
    using pbat::sim::vbd::multigrid::Restriction;
    using pbat::sim::vbd::multigrid::VolumeMesh;
    pyb::class_<Restriction>(m, "Restriction")
        .def(
            pyb::init(
                [](Data const& problem, Mesh const& FM, Mesh const& CM, CageQuadrature const& CQ) {
                    VolumeMesh const* FMraw = FM.Raw<VolumeMesh>();
                    VolumeMesh const* CMraw = CM.Raw<VolumeMesh>();
                    if (FMraw == nullptr or CMraw == nullptr)
                        throw std::invalid_argument(
                            "Requested underlying MeshType that this Mesh does not hold.");
                    return Restriction(problem, *FMraw, *CMraw, CQ);
                }),
            pyb::arg("problem"),
            pyb::arg("fine_mesh"),
            pyb::arg("cage_mesh"),
            pyb::arg("cage_quadrature"))
        .def("apply", &Restriction::Apply, pyb::arg("iters"), pyb::arg("lf"), pyb::arg("lc"))
        .def_readwrite("efg", &Restriction::efg)
        .def_readwrite("Nfg", &Restriction::Nfg)
        .def_readwrite("xfg", &Restriction::xfg)
        .def_readwrite("rhog", &Restriction::rhog)
        .def_readwrite("mug", &Restriction::mug)
        .def_readwrite("lambdag", &Restriction::lambdag)
        .def_readwrite("Ncg", &Restriction::Ncg)
        .def_readwrite("GNcg", &Restriction::GNcg);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat