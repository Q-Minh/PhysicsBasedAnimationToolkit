#include "Prolongation.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/Mesh.h>
#include <pbat/sim/vbd/lod/Level.h>
#include <pbat/sim/vbd/lod/Prolongation.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindProlongation(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::py::fem::Mesh;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::VolumeMesh;
    using pbat::sim::vbd::lod::Level;
    using pbat::sim::vbd::lod::Prolongation;
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
        .def(
            "apply",
            [](Prolongation& P, Level const& lc, Level& lf) { P.Apply(lc, lf); },
            pyb::arg("lc"),
            pyb::arg("lf"))
        .def(
            "apply",
            [](Prolongation& P, Level const& lc, Data& lf) { P.Apply(lc, lf); },
            pyb::arg("lc"),
            pyb::arg("lf"))
        .def_readwrite("ec", &Prolongation::ec)
        .def_readwrite("Nc", &Prolongation::Nc);
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat