#include "Data.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Enums.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {

void BindData(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EInitializationStrategy;

    pyb::enum_<EInitializationStrategy>(m, "InitializationStrategy")
        .value("Position", EInitializationStrategy::Position)
        .value("Inertia", EInitializationStrategy::Inertia)
        .value("KineticEnergyMinimum", EInitializationStrategy::KineticEnergyMinimum)
        .value("AdaptiveVbd", EInitializationStrategy::AdaptiveVbd)
        .value("AdaptivePbat", EInitializationStrategy::AdaptivePbat)
        .export_values();

    pyb::class_<Data>(m, "Data")
        .def(pyb::init<>())
        .def(
            "with_volume_mesh",
            &Data::WithVolumeMesh,
            pyb::arg("X"),
            pyb::arg("T"),
            "Sets the FEM simulation mesh as array of 3x|#nodes| positions X and 4x|#elements| "
            "tetrahedral elements T.")
        .def(
            "with_surface_mesh",
            &Data::WithSurfaceMesh,
            pyb::arg("V"),
            pyb::arg("F"),
            "Sets the collision mesh as array of 1x|#collision vertices| indices V into positions "
            "X and 3x|#collision triangles| indices into X.")
        .def(
            "with_velocity",
            &Data::WithVelocity,
            pyb::arg("v"),
            "Sets the 3x|#nodes| initial velocity field at FEM nodes.")
        .def(
            "with_acceleration",
            &Data::WithAcceleration,
            pyb::arg("a"),
            "Sets the 3x|#nodes| external acceleration field at FEM nodes.")
        .def(
            "with_mass",
            &Data::WithMass,
            pyb::arg("m"),
            "Sets the |#nodes| array of lumped nodal masses.")
        .def(
            "with_quadrature",
            &Data::WithQuadrature,
            pyb::arg("wg"),
            pyb::arg("GP"),
            pyb::arg("lame"),
            "Sets the |#quad.pts.| array of quadrature points for the total elastic potential, "
            "including the 3x|4*#elements| array of element shape function gradients GP, and "
            "2x|#elements| array of Lame coefficients.")
        .def(
            "with_vertex_adjacency",
            &Data::WithVertexAdjacency,
            pyb::arg("GVGp"),
            pyb::arg("GVGg"),
            pyb::arg("GVGe"),
            pyb::arg("GVGilocal"),
            "Sets the graph of (vertex, quadrature point) edges in the compressed sparse format, "
            "where GVGp is the |#nodes+1| prefix array, GVGg yields the adjacent quadrature "
            "points, GVGe yields the adjacent elements containing the corresponding quadrature "
            "points and GVGilocal yields the local vertex index in the corresponding adjacent "
            "element.")
        .def(
            "with_partitions",
            &Data::WithPartitions,
            pyb::arg("partitions"),
            "Sets the independent vertex partitions for solver parallelization.")
        .def(
            "with_dirichlet_vertices",
            &Data::WithDirichletConstrainedVertices,
            pyb::arg("dbc"),
            pyb::arg("input_sorted")      = true,
            pyb::arg("partitions_sorted") = false,
            "Sets Dirichlet constrained vertices.")
        .def(
            "with_initialization_strategy",
            &Data::WithInitializationStrategy,
            pyb::arg("strategy"))
        .def("with_rayleigh_damping", &Data::WithRayleighDamping, pyb::arg("kD"))
        .def("with_collision_penalty", &Data::WithCollisionPenalty, pyb::arg("kC"))
        .def(
            "with_hessian_determinant_zero",
            &Data::WithHessianDeterminantZeroUnder,
            pyb::arg("zero"))
        .def("construct", &Data::Construct, pyb::arg("validate") = true)
        .def_readwrite("V", &Data::V)
        .def_readwrite("F", &Data::F)
        .def_readwrite("T", &Data::T)
        .def_readwrite("x", &Data::x)
        .def_readwrite("v", &Data::v)
        .def_readwrite("aext", &Data::aext)
        .def_readwrite("m", &Data::m)
        .def_readwrite("xt", &Data::xt)
        .def_readwrite("vt", &Data::vt)
        .def_readwrite("wg", &Data::wg)
        .def_readwrite("lame", &Data::lame)
        .def_readwrite("GVGp", &Data::GVGp)
        .def_readwrite("GVGg", &Data::GVGg)
        .def_readwrite("GVGe", &Data::GVGe)
        .def_readwrite("GVGilocal", &Data::GVGilocal)
        .def_readwrite("dbc", &Data::dbc)
        .def_readwrite("partitions", &Data::partitions)
        .def_readwrite("strategy", &Data::strategy)
        .def_readwrite("kD", &Data::kD)
        .def_readwrite("kC", &Data::kC)
        .def_readwrite("detH_zero", &Data::detHZero);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat