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
            "with_bodies",
            &Data::WithBodies,
            pyb::arg("B"),
            "Sets the body indices of each vertex.\n\n"
            "Args:\n"
            "    B (numpy.ndarray): 1x|#nodes| array of body indices.")
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
            "with_material",
            &Data::WithMaterial,
            pyb::arg("rhoe"),
            pyb::arg("mue"),
            pyb::arg("lambdae"),
            "Sets the |#elements| array of mass densities, |#elements| array of 1st Lame "
            "coefficients and |#elements| array of 2nd Lame coefficients.")
        .def(
            "with_dirichlet_vertices",
            &Data::WithDirichletConstrainedVertices,
            pyb::arg("dbc"),
            pyb::arg("muD")          = Scalar(1),
            pyb::arg("input_sorted") = true,
            "Sets Dirichlet constrained vertices.")
        .def(
            "with_vertex_coloring_strategy",
            &Data::WithVertexColoringStrategy,
            pyb::arg("ordering"),
            pyb::arg("selection"),
            "Sets the vertex coloring strategy to use.")
        .def(
            "with_initialization_strategy",
            &Data::WithInitializationStrategy,
            pyb::arg("strategy"))
        .def("with_rayleigh_damping", &Data::WithRayleighDamping, pyb::arg("kD"))
        .def("with_collision_penalty", &Data::WithCollisionPenalty, pyb::arg("kC"))
        .def(
            "with_active_set_update_frequency",
            &Data::WithActiveSetUpdateFrequency,
            pyb::arg("frequency"),
            "Sets the contact constraint active set update frequency in a given time step (i.e. "
            "update vertex-triangle contact pairs every 'frequency' substeps).")
        .def(
            "with_hessian_determinant_zero",
            &Data::WithHessianDeterminantZeroUnder,
            pyb::arg("zero"))
        .def("construct", &Data::Construct, pyb::arg("validate") = true)
        .def_readwrite("X", &Data::X)
        .def_readwrite("E", &Data::E)
        .def_readwrite("V", &Data::V)
        .def_readwrite("F", &Data::F)
        .def_readwrite("x", &Data::x)
        .def_readwrite("v", &Data::v)
        .def_readwrite("aext", &Data::aext)
        .def_readwrite("m", &Data::m)
        .def_readwrite("xt", &Data::xt)
        .def_readwrite("vt", &Data::vt)
        .def_readwrite("wg", &Data::wg)
        .def_readwrite("rhoe", &Data::rhoe)
        .def_readwrite("lame", &Data::lame)
        .def_readwrite("GVGp", &Data::GVGp)
        .def_readwrite("GVGe", &Data::GVGe)
        .def_readwrite("GVGilocal", &Data::GVGilocal)
        .def_readwrite("dbc", &Data::dbc)
        .def_readwrite("vertex_coloring_ordering", &Data::eOrdering)
        .def_readwrite("vertex_coloring_selection", &Data::eSelection)
        .def_readwrite("colors", &Data::colors)
        .def_readwrite("Pptr", &Data::Pptr)
        .def_readwrite("Padj", &Data::Padj)
        .def_readwrite("strategy", &Data::strategy)
        .def_readwrite("kD", &Data::kD)
        .def_readwrite("kC", &Data::kC)
        .def_readwrite("detH_zero", &Data::detHZero);
}

} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat