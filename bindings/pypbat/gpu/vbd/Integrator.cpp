#include "Integrator.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <pbat/gpu/Aliases.h>
#include <pbat/gpu/vbd/Integrator.h>
#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Enums.h>
#include <utility>

namespace pbat {
namespace py {
namespace gpu {
namespace vbd {

void BindIntegrator([[maybe_unused]] nanobind::module_& m)
{
    namespace nb = nanobind;
#ifdef PBAT_USE_CUDA

    using namespace pbat;
    using pbat::gpu::vbd::Integrator;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EInitializationStrategy;

    nb::class_<Integrator>(m, "Integrator")
        .def(
            nb::init<Data const&>(),
            nb::arg("data"),
            "Construct a VBD algorithm to run on the GPU using data.")
        .def(
            "step",
            &Integrator::Step,
            nb::arg("dt")         = static_cast<GpuScalar>(0.01),
            nb::arg("iterations") = GpuIndex{20},
            nb::arg("substeps")   = GpuIndex{1},
            "Integrate 1 time step.\n\n"
            "Args:\n"
            "    dt (float): Time step\n"
            "    iterations (int): Number of optimization iterations per substep\n"
            "    substeps (int): Number of substeps")
        .def_prop_rw(
            "x",
            &Integrator::GetPositions,
            &Integrator::SetPositions,
            "|#dims|x|#vertices| vertex positions")
        .def_prop_rw(
            "v",
            &Integrator::GetVelocities,
            &Integrator::SetVelocities,
            "|#dims|x|#vertices| vertex velocities")
        .def_prop_rw(
            "a",
            nullptr,
            &Integrator::SetExternalAcceleration,
            "|#dims|x|#vertices| vertex external accelerations")
        .def_prop_rw(
            "detH_residual",
            nullptr,
            &Integrator::SetNumericalZeroForHessianDeterminant,
            "Numerical zero used in Hessian determinant check for approximate singularity "
            "detection")
        .def_prop_rw(
            "kD",
            nullptr,
            &Integrator::SetRayleighDampingCoefficient,
            "Uniform Rayleigh damping coefficient on the mesh.")
        .def_prop_rw(
            "strategy",
            nullptr,
            &Integrator::SetInitializationStrategy,
            "VBD's time step minimization initialization strategy")
        .def_prop_rw(
            "gpu_block_size",
            nullptr,
            &Integrator::SetBlockSize,
            "Number of threads per GPU thread block used for time integration "
            "minimization.")
        .def_prop_rw(
            "scene_bounding_box",
            nullptr,
            [](Integrator& vbd,
               std::pair<Eigen::Vector<GpuScalar, 3> const&, Eigen::Vector<GpuScalar, 3> const&>
                   box) { vbd.SetSceneBoundingBox(box.first, box.second); },
            "Tuple of (min,max) scene bounding box extremities.");
#endif // PBAT_USE_CUDA
}

} // namespace vbd
} // namespace gpu
} // namespace py
} // namespace pbat