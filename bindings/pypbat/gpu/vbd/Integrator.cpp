#include "Integrator.h"

#include <pbat/gpu/Aliases.h>
#include <pbat/gpu/vbd/Vbd.h>
#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/Enums.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <utility>

namespace pbat {
namespace py {
namespace gpu {
namespace vbd {

void BindIntegrator([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;
#ifdef PBAT_USE_CUDA

    using namespace pbat;
    using pbat::gpu::vbd::Integrator;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::EInitializationStrategy;

    pyb::class_<Integrator>(m, "Integrator")
        .def(
            pyb::init<Data const&>(),
            pyb::arg("data"),
            "Construct a VBD algorithm to run on the GPU using data.")
        .def(
            "step",
            &Integrator::Step,
            pyb::arg("dt")         = GpuScalar{0.01},
            pyb::arg("iterations") = GpuIndex{20},
            pyb::arg("substeps")   = GpuIndex{1},
            "Integrate 1 time step.\n\n"
            "Args:\n"
            "    dt (float): Time step\n"
            "    iterations (int): Number of optimization iterations per substep\n"
            "    substeps (int): Number of substeps")
        .def(
            "traced_step",
            &Integrator::TracedStep,
            pyb::arg("dt")         = GpuScalar{0.01},
            pyb::arg("iterations") = GpuIndex{20},
            pyb::arg("substeps")   = GpuIndex{1},
            pyb::arg("t"),
            "Integrate 1 time step and trace the result to disk.\n"
            "The result is saved in the current working directory as matrix market files.\n"
            "Filenames follow the pattern {variable}.t.{timestep}.s.{substep}[.k.{iteration}].mtx\n"
            "\n"
            "Args:\n"
            "    dt (float): Time step. Defaults to 0.01.\n"
            "    iterations (int): Number of optimization iterations per substep. Defaults to 20.\n"
            "    substeps (int): Number of substeps. Defaults to 1.\n"
            "    t (int): Current time step")
        .def_property(
            "x",
            &Integrator::GetPositions,
            &Integrator::SetPositions,
            "|#dims|x|#vertices| vertex positions")
        .def_property(
            "v",
            &Integrator::GetVelocities,
            &Integrator::SetVelocities,
            "|#dims|x|#vertices| vertex velocities")
        .def_property(
            "a",
            nullptr,
            &Integrator::SetExternalAcceleration,
            "|#dims|x|#vertices| vertex external accelerations")
        .def_property(
            "detH_residual",
            nullptr,
            &Integrator::SetNumericalZeroForHessianDeterminant,
            "Numerical zero used in Hessian determinant check for approximate singularity "
            "detection")
        .def_property(
            "kD",
            nullptr,
            &Integrator::SetRayleighDampingCoefficient,
            "Uniform Rayleigh damping coefficient on the mesh.")
        .def_property(
            "strategy",
            nullptr,
            &Integrator::SetInitializationStrategy,
            "VBD's time step minimization initialization strategy")
        .def_property(
            "gpu_block_size",
            nullptr,
            &Integrator::SetBlockSize,
            "Number of threads per GPU thread block used for time integration "
            "minimization.")
        .def_property(
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