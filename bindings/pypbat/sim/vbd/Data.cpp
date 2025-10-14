#include "Data.h"

#include <nanobind/eigen/dense.h>
#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Enums.h>

namespace pbat::py::sim::vbd {

void BindData(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EAccelerationStrategy;
    using pbat::sim::vbd::EBroydenJacobianEstimate;
    using pbat::sim::vbd::EInitializationStrategy;

    nb::enum_<EInitializationStrategy>(m, "InitializationStrategy")
        .value("Position", EInitializationStrategy::Position)
        .value("Inertia", EInitializationStrategy::Inertia)
        .value("KineticEnergyMinimum", EInitializationStrategy::KineticEnergyMinimum)
        .value("AdaptiveVbd", EInitializationStrategy::AdaptiveVbd)
        .value("AdaptivePbat", EInitializationStrategy::AdaptivePbat)
        .export_values();

    nb::enum_<EAccelerationStrategy>(m, "AccelerationStrategy")
        .value("Base", EAccelerationStrategy::None)
        .value("Chebyshev", EAccelerationStrategy::Chebyshev)
        .value("Anderson", EAccelerationStrategy::Anderson)
        .value("Nesterov", EAccelerationStrategy::Nesterov)
        .export_values();

    nb::enum_<EBroydenJacobianEstimate>(m, "BroydenJacobianEstimate")
        .value("Identity", EBroydenJacobianEstimate::Identity)
        .value("DiagonalCauchySchwarz", EBroydenJacobianEstimate::DiagonalCauchySchwarz)
        .export_values();

    nb::class_<Data>(m, "Data")
        .def(nb::init<>())
        .def(
            "with_volume_mesh",
            &Data::WithVolumeMesh,
            nb::arg("X"),
            nb::arg("T"),
            nb::rv_policy::reference_internal,
            "Sets the FEM simulation mesh as array of 3x|#nodes| positions X and 4x|#elements| "
            "tetrahedral elements T.\n\n"
            "Args:\n"
            "    X (numpy.ndarray): 3x|#nodes| array of vertex positions\n"
            "    T (numpy.ndarray): 4x|#elements| array of tetrahedral elements\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_surface_mesh",
            &Data::WithSurfaceMesh,
            nb::arg("V"),
            nb::arg("F"),
            nb::rv_policy::reference_internal,
            "Sets the collision mesh as array of 1x|#collision vertices| indices V into positions "
            "X and 3x|#collision triangles| indices into X.\n\n"
            "Args:\n"
            "    V (numpy.ndarray): 1x|#collision vertices| array of vertex indices\n"
            "    F (numpy.ndarray): 3x|#collision triangles| array of triangle indices\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_bodies",
            &Data::WithBodies,
            nb::arg("B"),
            nb::rv_policy::reference_internal,
            "Sets the body indices of each vertex.\n\n"
            "Args:\n"
            "    B (numpy.ndarray): 1x|#nodes| array of body indices.\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_velocity",
            &Data::WithVelocity,
            nb::arg("v"),
            nb::rv_policy::reference_internal,
            "Sets the 3x|#nodes| initial velocity field at FEM nodes.\n\n"
            "Args:\n"
            "    v (numpy.ndarray): 3x|#nodes| array of initial velocities.\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_acceleration",
            &Data::WithAcceleration,
            nb::arg("a"),
            nb::rv_policy::reference_internal,
            "Sets the 3x|#nodes| external acceleration field at FEM nodes.\n\n"
            "Args:\n"
            "    a (numpy.ndarray): 3x|#nodes| array of external accelerations.\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_material",
            &Data::WithMaterial,
            nb::arg("rhoe"),
            nb::arg("mue"),
            nb::arg("lambdae"),
            nb::rv_policy::reference_internal,
            "Sets the |#elements| array of mass densities, |#elements| array of 1st Lame "
            "coefficients and |#elements| array of 2nd Lame coefficients.\n\n"
            "Args:\n"
            "    rhoe (numpy.ndarray): |#elements| array of mass densities\n"
            "    mue (numpy.ndarray): |#elements| array of 1st Lame coefficients\n"
            "    lambdae (numpy.ndarray): |#elements| array of 2nd Lame coefficients\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_dirichlet_vertices",
            &Data::WithDirichletConstrainedVertices,
            nb::arg("dbc"),
            nb::arg("muD")          = Scalar(1),
            nb::arg("input_sorted") = true,
            nb::rv_policy::reference_internal,
            "Sets Dirichlet constrained vertices.\n\n"
            "Args:\n"
            "    dbc (numpy.ndarray): Dirichlet constrained vertices\n"
            "    muD (float): Dirichlet penalty\n"
            "    input_sorted (bool): True if dbc is sorted\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_vertex_coloring_strategy",
            &Data::WithVertexColoringStrategy,
            nb::arg("ordering"),
            nb::arg("selection"),
            nb::rv_policy::reference_internal,
            "Sets the vertex coloring strategy to use.\n\n"
            "Args:\n"
            "    ordering (int): Vertex visit order\n"
            "    selection (int): Color selection strategy\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_initialization_strategy",
            &Data::WithInitializationStrategy,
            nb::arg("strategy"),
            nb::rv_policy::reference_internal,
            "Sets the non-linear optimization initialization strategy.\n\n"
            "Args:\n"
            "    strategy (int): Initialization strategy\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_rayleigh_damping",
            &Data::WithRayleighDamping,
            nb::arg("kD"),
            nb::rv_policy::reference_internal,
            "Sets the Rayleigh damping coefficient.\n\n"
            "Args:\n"
            "    kD (float): Damping coefficient\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_contact_parameters",
            &Data::WithContactParameters,
            nb::arg("muC"),
            nb::arg("muF"),
            nb::arg("epsv"),
            nb::rv_policy::reference_internal,
            "Sets the variational contact model's parameters.\n\n"
            "Args:\n"
            "    muC (float): Collision penalty\n"
            "    muF (float): Friction coefficient\n"
            "    epsv (float): IPC's relative velocity threshold for static to dynamic friction's "
            "smooth transition.\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_active_set_update_frequency",
            &Data::WithActiveSetUpdateFrequency,
            nb::arg("frequency"),
            nb::rv_policy::reference_internal,
            "Sets the contact constraint active set update frequency in a given time step (i.e. "
            "update vertex-triangle contact pairs every 'frequency' substeps).\n\n"
            "Args:\n"
            "    frequency (int): Active set update frequency\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_hessian_determinant_zero",
            &Data::WithHessianDeterminantZeroUnder,
            nb::arg("zero"),
            nb::rv_policy::reference_internal,
            "Sets the numerical zero used in 'singular' hessian determinant check.\n\n"
            "Args:\n"
            "    zero (float): Numerical zero\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_chebyshev_acceleration",
            &Data::WithChebyshevAcceleration,
            nb::arg("rho"),
            nb::rv_policy::reference_internal,
            "Use Chebyshev semi-iterative method's\n\n"
            "Args:\n"
            "    rho (float): Estimated spectral radius. rho must be in (0, 1).\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_anderson_acceleration",
            &Data::WithAndersonAcceleration,
            nb::arg("window_size"),
            nb::rv_policy::reference_internal,
            "Use Anderson acceleration\n\n"
            "Args:\n"
            "    window (int): Number of past iterates to use in Anderson acceleration.\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_broyden_acceleration",
            &Data::WithBroydenMethod,
            nb::arg("window_size"),
            nb::arg("jacobian_estimate") = EBroydenJacobianEstimate::Identity,
            nb::arg("broyden_beta")      = Scalar{1},
            nb::rv_policy::reference_internal,
            "Use Broyden acceleration\n\n"
            "Args:\n"
            "    window (int): Number of past iterates to use in Broyden acceleration.\n\n"
            "    jacobian_estimate (BroydenJacobianEstimate): Broyden Jacobian estimate "
            "strategy.\n\n"
            "    broyden_beta (float): Broyden Cauchy-Schwarz scaling factor.\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "with_nesterov_acceleration",
            &Data::WithNesterovAcceleration,
            nb::arg("L"),
            nb::arg("start"),
            nb::rv_policy::reference_internal,
            "Use Nesterov acceleration\n\n"
            "Args:\n"
            "    L (float): Estimated gradient Lipschitz constant\n"
            "    start (int): Iteration to start Nesterov acceleration\n\n"
            "Returns:\n"
            "    Data: self")
        .def(
            "construct",
            &Data::Construct,
            nb::arg("validate") = true,
            nb::rv_policy::reference_internal,
            "Constructs the VBD data.\n\n"
            "Args:\n"
            "    validate (bool): True to validate the data\n\n"
            "Returns:\n"
            "    Data: self")
        .def_rw("X", &Data::X, "3x|#nodes| array of vertex positions")
        .def_rw("E", &Data::E, "4x|#elements| array of tetrahedral elements")
        .def_rw("V", &Data::V, "1x|#collision vertices| array of vertex indices")
        .def_rw("F", &Data::F, "3x|#collision triangles| array of triangle indices")
        .def_rw("x", &Data::x, "3x|#nodes| array of vertex positions")
        .def_rw("v", &Data::v, "3x|#nodes| array of vertex velocities")
        .def_rw("aext", &Data::aext, "3x|#nodes| array of external accelerations")
        .def_rw("m", &Data::m, "|#nodes| array of vertex mass")
        .def_rw("xt", &Data::xt, "3x|#nodes| array of vertex positions at time t")
        .def_rw("vt", &Data::vt, "3x|#nodes| array of vertex velocities at time t")
        .def_rw("wg", &Data::wg, "|#elements| array of element quadrature weights")
        .def_rw("rhoe", &Data::rhoe, "|#elements| array of mass densities")
        .def_rw("lame", &Data::lame, "2x|#elements| array of Lame coefficients")
        .def_rw("GVGp", &Data::GVGp, "|#nodes|+1 prefixes into GVGg")
        .def_rw("GVGe", &Data::GVGe, "|#vertex-elems edges| element indices")
        .def_rw("GVGilocal", &Data::GVGilocal, "|#vertex-elems edges| local indices")
        .def_rw("dbc", &Data::dbc, "Dirichlet constrained vertices")
        .def_rw("vertex_coloring_ordering", &Data::eOrdering, "Vertex visit order")
        .def_rw("vertex_coloring_selection", &Data::eSelection, "Color selection strategy")
        .def_rw("colors", &Data::colors, "|#nodes| array of vertex colors")
        .def_rw("Pptr", &Data::Pptr, "|#nodes|+1 prefixes into Padj")
        .def_rw("Padj", &Data::Padj, "|#vertex-vertex edges| vertex indices")
        .def_rw("strategy", &Data::strategy, "BCD optimization initialization strategy")
        .def_rw("kD", &Data::kD, "Uniform damping coefficient")
        .def_rw("muC", &Data::muC, "Uniform collision penalty")
        .def_rw("muF", &Data::muF, "Uniform friction coefficient")
        .def_rw("epsv", &Data::epsv, "IPC's relative velocity threshold for smooth transition")
        .def_rw("detH_zero", &Data::detHZero, "Numerical zero for hessian pseudo-singularity check")
        .def_rw("accelerator", &Data::eAcceleration, "Acceleration strategy")
        .def_rw("rho", &Data::rho, "Chebyshev acceleration estimated spectral radius")
        .def_rw("window_size", &Data::mWindowSize, "Anderson acceleration window size")
        .def_rw(
            "jacobian_estimate",
            &Data::eBroydenJacobianEstimate,
            "Broyden Jacobian estimate strategy");
}

} // namespace pbat::py::sim::vbd