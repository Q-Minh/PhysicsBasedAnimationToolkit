#include "Data.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Enums.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace pbat::py::sim::vbd {

void BindData(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EAccelerationStrategy;
    using pbat::sim::vbd::EInitializationStrategy;

    pyb::enum_<EInitializationStrategy>(m, "InitializationStrategy")
        .value("Position", EInitializationStrategy::Position)
        .value("Inertia", EInitializationStrategy::Inertia)
        .value("KineticEnergyMinimum", EInitializationStrategy::KineticEnergyMinimum)
        .value("AdaptiveVbd", EInitializationStrategy::AdaptiveVbd)
        .value("AdaptivePbat", EInitializationStrategy::AdaptivePbat)
        .export_values();

    pyb::enum_<EAccelerationStrategy>(m, "AccelerationStrategy")
        .value("None", EAccelerationStrategy::None)
        .value("Chebyshev", EAccelerationStrategy::Chebyshev)
        .value("TrustRegion", EAccelerationStrategy::TrustRegion)
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
        .def(
            "with_contact_parameters",
            &Data::WithContactParameters,
            pyb::arg("muC"),
            pyb::arg("muF"),
            pyb::arg("epsv"),
            "Sets the variational contact model's parameters.\n\n"
            "Args:\n"
            "    muC (float): Collision penalty\n"
            "    muF (float): Friction coefficient\n"
            "    epsv (float): IPC's relative velocity threshold for static to dynamic friction's "
            "smooth transition.")
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
        .def(
            "with_chebyshev_acceleration",
            &Data::WithChebyshevAcceleration,
            pyb::arg("rho"),
            "Use Chebyshev semi-iterative method's\n\n"
            "Args:\n"
            "    rho (float): Estimated spectral radius. rho must be in (0, 1).")
        .def(
            "with_trust_region_acceleration",
            &Data::WithTrustRegionAcceleration,
            pyb::arg("eta"),
            pyb::arg("tau"),
            pyb::arg("curved"),
            "Use Trust Region acceleration\n\n"
            "Args:\n"
            "    eta (float): Energy reduction accuracy threshold\n"
            "    tau (float): Trust Region radius increase factor\n"
            "    curved (bool): Use curved accelerated path, otherwise use linear path.")
        .def("construct", &Data::Construct, pyb::arg("validate") = true)
        .def_readwrite("X", &Data::X, "3x|#nodes| array of vertex positions")
        .def_readwrite("E", &Data::E, "4x|#elements| array of tetrahedral elements")
        .def_readwrite("V", &Data::V, "1x|#collision vertices| array of vertex indices")
        .def_readwrite("F", &Data::F, "3x|#collision triangles| array of triangle indices")
        .def_readwrite("x", &Data::x, "3x|#nodes| array of vertex positions")
        .def_readwrite("v", &Data::v, "3x|#nodes| array of vertex velocities")
        .def_readwrite("aext", &Data::aext, "3x|#nodes| array of external accelerations")
        .def_readwrite("m", &Data::m, "|#nodes| array of vertex mass")
        .def_readwrite("xt", &Data::xt, "3x|#nodes| array of vertex positions at time t")
        .def_readwrite("vt", &Data::vt, "3x|#nodes| array of vertex velocities at time t")
        .def_readwrite("wg", &Data::wg, "|#elements| array of element quadrature weights")
        .def_readwrite("rhoe", &Data::rhoe, "|#elements| array of mass densities")
        .def_readwrite("lame", &Data::lame, "2x|#elements| array of Lame coefficients")
        .def_readwrite("GVGp", &Data::GVGp, "|#nodes|+1 prefixes into GVGg")
        .def_readwrite("GVGe", &Data::GVGe, "|#vertex-elems edges| element indices")
        .def_readwrite("GVGilocal", &Data::GVGilocal, "|#vertex-elems edges| local indices")
        .def_readwrite("dbc", &Data::dbc, "Dirichlet constrained vertices")
        .def_readwrite("vertex_coloring_ordering", &Data::eOrdering, "Vertex visit order")
        .def_readwrite("vertex_coloring_selection", &Data::eSelection, "Color selection strategy")
        .def_readwrite("colors", &Data::colors, "|#nodes| array of vertex colors")
        .def_readwrite("Pptr", &Data::Pptr, "|#nodes|+1 prefixes into Padj")
        .def_readwrite("Padj", &Data::Padj, "|#vertex-vertex edges| vertex indices")
        .def_readwrite("strategy", &Data::strategy, "BCD optimization initialization strategy")
        .def_readwrite("kD", &Data::kD, "Uniform damping coefficient")
        .def_readwrite("muC", &Data::muC, "Uniform collision penalty")
        .def_readwrite("muF", &Data::muF, "Uniform friction coefficient")
        .def_readwrite(
            "epsv",
            &Data::epsv,
            "IPC's relative velocity threshold for smooth transition")
        .def_readwrite(
            "detH_zero",
            &Data::detHZero,
            "Numerical zero for hessian pseudo-singularity check")
        .def_readwrite("accelerator", &Data::eAcceleration, "Acceleration strategy")
        .def_readwrite("rho", &Data::rho, "Chebyshev acceleration estimated spectral radius")
        .def_readwrite("eta", &Data::eta, "Trust Region energy reduction accuracy threshold")
        .def_readwrite("tau", &Data::tau, "Trust Region radius increase factor")
        .def_readwrite(
            "curved",
            &Data::bCurved,
            "Use curved accelerated path for Trust Region acceleration");
}

} // namespace pbat::py::sim::vbd