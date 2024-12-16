#include "Level.h"

#include "pypbat/fem/Mesh.h"

#include <pbat/sim/vbd/Mesh.h>
#include <pbat/sim/vbd/lod/Level.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace lod {

void BindLevel(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::VolumeMesh;
    using pbat::sim::vbd::lod::ECageQuadratureStrategy;
    using pbat::sim::vbd::lod::Level;

    pyb::class_<Level>(m, "Level")
        .def(
            pyb::init([](pbat::py::fem::Mesh const& CM) {
                VolumeMesh const* CMraw = CM.Raw<VolumeMesh>();
                if (CMraw == nullptr)
                    throw std::invalid_argument(
                        "Requested underlying MeshType that this Mesh does not hold.");
                return Level(*CMraw);
            }),
            pyb::arg("cage_mesh"))
        .def(
            "with_cage_quadrature",
            &Level::WithCageQuadrature,
            pyb::arg("problem"),
            pyb::arg("params"))
        .def("with_dirichlet_quadrature", &Level::WithDirichletQuadrature, pyb::arg("problem"))
        .def("with_momentum_energy", &Level::WithMomentumEnergy, pyb::arg("problem"))
        .def("with_elastic_energy", &Level::WithElasticEnergy, pyb::arg("problem"))
        .def("with_dirichlet_energy", &Level::WithDirichletEnergy, pyb::arg("problem"))
        .def_readwrite("x", &Level::x)
        .def_property(
            "X",
            [](Level const& l) { return l.mesh.X; },
            [](Level& l, Eigen::Ref<MatrixX const> const& X) { l.mesh.X = X; })
        .def_property(
            "E",
            [](Level const& l) { return l.mesh.E; },
            [](Level& l, Eigen::Ref<IndexMatrixX const> const& E) { l.mesh.E = E; })
        .def_readwrite("Qcage", &Level::Qcage)
        .def_readwrite("Qdirichlet", &Level::Qdirichlet)
        .def_readwrite("Ekinetic", &Level::Ekinetic)
        .def_readwrite("Epotential", &Level::Epotential)
        .def_readwrite("Edirichlet", &Level::Edirichlet);
}

} // namespace lod
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat