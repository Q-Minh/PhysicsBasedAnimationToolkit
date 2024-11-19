#include "Data.h"

#include <pbat/sim/xpbd/Data.h>
#include <pbat/sim/xpbd/Enums.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void BindData(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::xpbd::Data;
    using pbat::sim::xpbd::EConstraint;

    pyb::enum_<EConstraint>(m, "Constraint")
        .value("StableNeoHookean", EConstraint::StableNeoHookean)
        .value("Collision", EConstraint::Collision)
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
            "Sets the collision mesh as array of |#collision particles| indices V into positions "
            "X and 3x|#collision triangles| indices into X.")
        .def(
            "with_bodies",
            &Data::WithBodies,
            pyb::arg("BV"),
            "Sets |#particles| array of bodies associated with particle indices.")
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
            "with_mass_inverse",
            &Data::WithMassInverse,
            pyb::arg("minv"),
            "Sets the |#nodes| array of lumped nodal inverse masses.")
        .def(
            "with_elastic_material",
            &Data::WithElasticMaterial,
            pyb::arg("lame"),
            "Sets the 2x|#tetrahedra| array of Lame coefficients.")
        .def(
            "with_friction_coefficients",
            &Data::WithFrictionCoefficients,
            pyb::arg("muS"),
            pyb::arg("muD"),
            "Sets the static and dynamic friction coefficients.")
        .def(
            "with_collision_penalties",
            &Data::WithCollisionPenalties,
            pyb::arg("muV"),
            "Sets the |#collision vertices| array of collision penalty coefficients.")
        .def(
            "with_compliance",
            &Data::WithCompliance,
            pyb::arg("alpha"),
            pyb::arg("constraint"),
            "Sets the constraint compliance for the given constraint type.")
        .def(
            "with_damping",
            &Data::WithDamping,
            pyb::arg("beta"),
            pyb::arg("constraint"),
            "Sets the constraint damping for the given constraint type.")
        .def(
            "with_partitions",
            &Data::WithPartitions,
            pyb::arg("Pptr"),
            pyb::arg("Padj"),
            "Sets the independent constraint partitions for solver parallelization.")
        .def(
            "with_cluster_partitions",
            &Data::WithClusterPartitions,
            pyb::arg("SGptr"),
            pyb::arg("SGadj"),
            pyb::arg("Cptr"),
            pyb::arg("Cadj"),
            "Sets the independent constraint cluster partitions for clustered solver "
            "parallelization.")
        .def(
            "with_dirichlet_vertices",
            &Data::WithDirichletConstrainedVertices,
            pyb::arg("dbc"),
            "Sets Dirichlet constrained vertices.")
        .def("construct", &Data::Construct, pyb::arg("validate") = true)
        .def_readwrite("V", &Data::V)
        .def_readwrite("F", &Data::F)
        .def_readwrite("T", &Data::T)
        .def_readwrite("x", &Data::x)
        .def_readwrite("v", &Data::v)
        .def_readwrite("aext", &Data::aext)
        .def_readwrite("minv", &Data::minv)
        .def_readwrite("xt", &Data::xt)
        .def_readwrite("lame", &Data::lame)
        .def_readwrite("DmInv", &Data::DmInv)
        .def_readwrite("gammaSNH", &Data::gammaSNH)
        .def_readwrite("muS", &Data::muS)
        .def_readwrite("muD", &Data::muD)
        .def_readwrite("alpha", &Data::alpha)
        .def_readwrite("beta", &Data::beta)
        .def_readwrite("lambda", &Data::lambda)
        .def_readwrite("dbc", &Data::dbc)
        .def_readwrite("partitions_ptr", &Data::Pptr)
        .def_readwrite("partitions_adj", &Data::Padj);
}

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat