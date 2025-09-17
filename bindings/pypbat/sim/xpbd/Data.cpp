#include "Data.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <pbat/sim/xpbd/Data.h>
#include <pbat/sim/xpbd/Enums.h>

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void BindData(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::sim::xpbd::Data;
    using pbat::sim::xpbd::EConstraint;

    nb::enum_<EConstraint>(m, "Constraint")
        .value("StableNeoHookean", EConstraint::StableNeoHookean)
        .value("Collision", EConstraint::Collision)
        .export_values();

    nb::class_<Data>(m, "Data")
        .def(nb::init<>())
        .def(
            "with_volume_mesh",
            &Data::WithVolumeMesh,
            nb::arg("X"),
            nb::arg("T"),
            "Sets the FEM simulation mesh as array of 3x|#nodes| positions X and 4x|#elements| "
            "tetrahedral elements T.")
        .def(
            "with_surface_mesh",
            &Data::WithSurfaceMesh,
            nb::arg("V"),
            nb::arg("F"),
            "Sets the collision mesh as array of |#collision particles| indices V into positions "
            "X and 3x|#collision triangles| indices into X.")
        .def(
            "with_bodies",
            &Data::WithBodies,
            nb::arg("BV"),
            "Sets |#particles| array of bodies associated with particle indices.")
        .def(
            "with_velocity",
            &Data::WithVelocity,
            nb::arg("v"),
            "Sets the 3x|#nodes| initial velocity field at FEM nodes.")
        .def(
            "with_acceleration",
            &Data::WithAcceleration,
            nb::arg("a"),
            "Sets the 3x|#nodes| external acceleration field at FEM nodes.")
        .def(
            "with_mass_inverse",
            &Data::WithMassInverse,
            nb::arg("minv"),
            "Sets the |#nodes| array of lumped nodal inverse masses.")
        .def(
            "with_elastic_material",
            &Data::WithElasticMaterial,
            nb::arg("lame"),
            "Sets the 2x|#tetrahedra| array of Lame coefficients.")
        .def(
            "with_friction_coefficients",
            &Data::WithFrictionCoefficients,
            nb::arg("muS"),
            nb::arg("muD"),
            "Sets the static and dynamic friction coefficients.")
        .def(
            "with_collision_penalties",
            &Data::WithCollisionPenalties,
            nb::arg("muV"),
            "Sets the |#collision vertices| array of collision penalty coefficients.")
        .def(
            "with_active_set_update_frequency",
            &Data::WithActiveSetUpdateFrequency,
            nb::arg("frequency"),
            "Sets the contact constraint active set update frequency.")
        .def(
            "with_compliance",
            &Data::WithCompliance,
            nb::arg("alpha"),
            nb::arg("constraint"),
            "Sets the constraint compliance for the given constraint type.")
        .def(
            "with_damping",
            &Data::WithDamping,
            nb::arg("beta"),
            nb::arg("constraint"),
            "Sets the constraint damping for the given constraint type.")
        .def(
            "with_partitions",
            &Data::WithPartitions,
            nb::arg("Pptr"),
            nb::arg("Padj"),
            "Sets the independent constraint partitions for solver parallelization.")
        .def(
            "with_cluster_partitions",
            &Data::WithClusterPartitions,
            nb::arg("SGptr"),
            nb::arg("SGadj"),
            nb::arg("Cptr"),
            nb::arg("Cadj"),
            "Sets the independent constraint cluster partitions for clustered solver "
            "parallelization.")
        .def(
            "with_dirichlet_vertices",
            &Data::WithDirichletConstrainedVertices,
            nb::arg("dbc"),
            "Sets Dirichlet constrained vertices.")
        .def("construct", &Data::Construct, nb::arg("validate") = true)
        .def_rw("V", &Data::V)
        .def_rw("F", &Data::F)
        .def_rw("T", &Data::T)
        .def_rw("x", &Data::x)
        .def_rw("v", &Data::v)
        .def_rw("aext", &Data::aext)
        .def_rw("minv", &Data::minv)
        .def_rw("xt", &Data::xt)
        .def_rw("lame", &Data::lame)
        .def_rw("DmInv", &Data::DmInv)
        .def_rw("gammaSNH", &Data::gammaSNH)
        .def_rw("muS", &Data::muS)
        .def_rw("muD", &Data::muD)
        .def_rw("alpha", &Data::alpha)
        .def_rw("beta", &Data::beta)
        .def_rw("lambda", &Data::lambda)
        .def_rw("dbc", &Data::dbc)
        .def_rw("partitions_ptr", &Data::Pptr)
        .def_rw("partitions_adj", &Data::Padj);
}

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat