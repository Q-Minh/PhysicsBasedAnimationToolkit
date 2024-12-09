#include "MomentumEnergy.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/multigrid/MomentumEnergy.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindMomentumEnergy(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::multigrid::CageQuadrature;
    using pbat::sim::vbd::multigrid::MomentumEnergy;
    using pbat::sim::vbd::multigrid::VolumeMesh;
    pyb::class_<MomentumEnergy>(m, "MomentumEnergy")
        .def(
            pyb::init(
                [](Data const& problem, pbat::py::fem::Mesh const& CM, CageQuadrature const& CQ) {
                    VolumeMesh const* CMraw = CM.Raw<VolumeMesh>();
                    if (CMraw == nullptr)
                        throw std::invalid_argument(
                            "Requested underlying MeshType that this Mesh does not hold.");
                    return MomentumEnergy(problem, *CMraw, CQ);
                }),
            pyb::arg("problem"),
            pyb::arg("cage_mesh"),
            pyb::arg("cage_quadrature"))
        .def("update_inertial_target_positions", &MomentumEnergy::UpdateInertialTargetPositions)
        .def_readwrite("xtildeg", &MomentumEnergy::xtildeg)
        .def_readwrite("erg", &MomentumEnergy::erg)
        .def_readwrite("Nrg", &MomentumEnergy::Nrg)
        .def_readwrite("Ncg", &MomentumEnergy::Ncg)
        .def_readwrite("rhog", &MomentumEnergy::rhog);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat