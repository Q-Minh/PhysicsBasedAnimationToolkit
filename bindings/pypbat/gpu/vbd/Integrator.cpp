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
            pyb::arg("rho")        = GpuScalar{1},
            "Integrate 1 time step in time using a uniform time discretization dt. Use iterations "
            "iterations for the time integration minimization. Substepping can be specified in "
            "substeps, such that substeps*iterations total iterations will be executed. rho < 1 "
            "yields a spectral radius estimate for Chebyshev semi-iterative method acceleration, "
            "inactive if rho >= 1.")
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
        .def_property("m", nullptr, &Integrator::SetMass, "|#vertices| lumped masses")
        .def_property(
            "wg",
            nullptr,
            &Integrator::SetQuadratureWeights,
            "|#elements| array of quadrature weights")
        .def_property(
            "GNe",
            nullptr,
            &Integrator::SetShapeFunctionGradients,
            "4x3x|#elements| array of shape function gradients, stored column-wise (i.e. the 4x3 "
            "element shape function gradient matrices are flattened in column-major format)")
        .def_property(
            "lame",
            nullptr,
            &Integrator::SetLameCoefficients,
            "2x|#elements| Lame coefficients")
        .def_property(
            "detH_residual",
            nullptr,
            &Integrator::SetNumericalZeroForHessianDeterminant,
            "Numerical zero used in Hessian determinant check for approximate singularity "
            "detection")
        .def_property(
            "GVT",
            nullptr,
            [](Integrator& vbd,
               std::tuple<
                   Eigen::Ref<GpuIndexVectorX const>,
                   Eigen::Ref<GpuIndexVectorX const>,
                   Eigen::Ref<GpuIndexVectorX const>> const& GVT) {
                vbd.SetVertexTetrahedronAdjacencyList(
                    std::get<0>(GVT),
                    std::get<1>(GVT),
                    std::get<2>(GVT));
            },
            "3-tuple (prefix,neighbours,data) representing the compressed column storage graph "
            "representation of the vertex-tetrahedron adjacency list. The data property yields the "
            "local vertex index associated with a pair (i,e) of vertex i adjacent to element e.")
        .def_property(
            "kD",
            nullptr,
            &Integrator::SetRayleighDampingCoefficient,
            "Uniform Rayleigh damping coefficient on the mesh.")
        .def_property(
            "partitions",
            nullptr,
            [](Integrator& vbd,
               std::pair<Eigen::Ref<GpuIndexVectorX const>, Eigen::Ref<GpuIndexVectorX const>>
                   partitions) { vbd.SetVertexPartitions(partitions.first, partitions.second); },
            "Vertex partitions for the parallel time integration minimization solve as list of "
            "lists of vertex indices")
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