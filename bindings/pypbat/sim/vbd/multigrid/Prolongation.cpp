#include "Prolongation.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/multigrid/Level.h>
#include <pbat/sim/vbd/multigrid/Mesh.h>
#include <pbat/sim/vbd/multigrid/Prolongation.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindProlongation(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::py::fem::Mesh;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::multigrid::Prolongation;
    using pbat::sim::vbd::multigrid::VolumeMesh;
    pyb::class_<Prolongation>(m, "Prolongation")
        .def(
            pyb::init([](Mesh const& FM, Mesh const& CM) {
                VolumeMesh const* FMraw = FM.Raw<VolumeMesh>();
                VolumeMesh const* CMraw = CM.Raw<VolumeMesh>();
                if (FMraw == nullptr or CMraw == nullptr)
                    throw std::invalid_argument(
                        "Requested underlying MeshType that this Mesh does not hold.");
                return Prolongation(*FMraw, *CMraw);
            }),
            pyb::arg("fine_mesh"),
            pyb::arg("cage_mesh"))
        .def("apply", &Prolongation::Apply, pyb::arg("lc"), pyb::arg("lf"))
        .def_readwrite("ec", &Prolongation::ec)
        .def_readwrite("Nc", &Prolongation::Nc);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat