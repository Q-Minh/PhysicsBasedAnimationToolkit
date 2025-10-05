#include "Integrator.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <pbat/gpu/Aliases.h>
#include <pbat/gpu/xpbd/Integrator.h>
#include <pbat/sim/xpbd/Data.h>
#include <pbat/sim/xpbd/Enums.h>
#include <utility>

namespace pbat {
namespace py {
namespace gpu {
namespace xpbd {

void BindIntegrator([[maybe_unused]] nanobind::module_& m)
{
    namespace nb = nanobind;
#ifdef PBAT_USE_CUDA

    using namespace pbat;
    using pbat::gpu::xpbd::Integrator;
    using pbat::sim::xpbd::Data;
    using pbat::sim::xpbd::EConstraint;

    nb::class_<Integrator>(m, "Integrator")
        .def(
            nb::init<Data const&>(),
            nb::arg("data"),
            "Constructs an XPBD algorithm with the given data, where "
            "max_vertex_tetrahedron_overlaps specifies the size of memory preallocated for "
            "vertex-tetrahedron overlaps detected in the broad phase. max_vertex_triangle_contacts "
            "specifies maximum number of collision constraints that will be supported.")
        .def(
            "step",
            &Integrator::Step,
            nb::arg("dt")         = 0.01f,
            nb::arg("iterations") = 10,
            nb::arg("substeps")   = 5,
            "Integrate 1 time step in time")
        .def_prop_rw(
            "x",
            &Integrator::Positions,
            &Integrator::SetPositions,
            "|#dims|x|#particles| particle positions")
        .def_prop_rw(
            "v",
            nullptr,
            &Integrator::SetVelocities,
            "|#dims|x|#particles| particle velocities")
        .def(
            "set_compliance",
            &Integrator::SetCompliance,
            nb::arg("alpha"),
            nb::arg("constraint_type"),
            "Set the |#lagrange multiplier per constraint|x|#constraint of type eConstraint| "
            "constraint compliances for the given constraint type")
        .def_prop_rw(
            "mu",
            nullptr,
            [](Integrator& xpbd, std::pair<GpuScalar, GpuScalar> mu) {
                xpbd.SetFrictionCoefficients(mu.first, mu.second);
            },
            "Tuple of static and dynamic friction coefficients (muS, muK).")
        .def_prop_rw(
            "scene_bounding_box",
            nullptr,
            [](Integrator& xpbd,
               std::pair<Eigen::Vector<GpuScalar, 3> const&, Eigen::Vector<GpuScalar, 3> const&>
                   box) { xpbd.SetSceneBoundingBox(box.first, box.second); },
            "Tuple of (min,max) scene bounding box extremities.");
#endif // PBAT_USE_CUDA
}

} // namespace xpbd
} // namespace gpu
} // namespace py
} // namespace pbat