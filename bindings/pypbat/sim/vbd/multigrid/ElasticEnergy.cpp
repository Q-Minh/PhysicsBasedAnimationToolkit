#include "ElasticEnergy.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/multigrid/ElasticEnergy.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindElasticEnergy(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::multigrid::CageQuadrature;
    using pbat::sim::vbd::multigrid::ElasticEnergy;
    using pbat::sim::vbd::multigrid::VolumeMesh;
    pyb::class_<ElasticEnergy>(m, "ElasticEnergy")
        .def(
            pyb::init(
                [](Data const& problem, pbat::py::fem::Mesh const& CM, CageQuadrature const& CQ) {
                    VolumeMesh const* CMraw = CM.Raw<VolumeMesh>();
                    if (CMraw == nullptr)
                        throw std::invalid_argument(
                            "Requested underlying MeshType that this Mesh does not hold.");
                    return ElasticEnergy(problem, *CMraw, CQ);
                }),
            pyb::arg("problem"),
            pyb::arg("cage_mesh"),
            pyb::arg("cage_quadrature"))
        .def_readwrite("mug", &ElasticEnergy::mug)
        .def_readwrite("lambdag", &ElasticEnergy::lambdag)
        .def_readwrite("GNg", &ElasticEnergy::GNg);
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat