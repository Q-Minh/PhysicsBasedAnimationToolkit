#include "Integrator.h"

#include <pbat/gpu/Aliases.h>
#include <pbat/gpu/vbd/Vbd.h>
#include <pbat/profiling/Profiling.h>
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
            pyb::init([](Data const& data) {
                return pbat::profiling::Profile("pbat.gpu.vbd.Integrator.Construct", [&]() {
                    Integrator vbd(data);
                    return vbd;
                });
            }),
            pyb::arg("data"),
            "Construct a VBD algorithm to run on the GPU using data.")
        .def(
            "step",
            [](Integrator& vbd,
               GpuScalar dt,
               GpuIndex iterations,
               GpuIndex substeps,
               GpuScalar rho) {
                pbat::profiling::Profile("pbat.gpu.vbd.Integrator.Step", [&]() {
                    vbd.Step(dt, iterations, substeps, rho);
                });
            },
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
            [](Integrator const& vbd) {
                return pbat::profiling::Profile("pbat.gpu.vbd.Integrator.GetPositions", [&]() {
                    GpuMatrixX X = vbd.GetPositions();
                    return X;
                });
            },
            [](Integrator& vbd, Eigen::Ref<GpuMatrixX const> const& X) {
                pbat::profiling::Profile("pbat.gpu.vbd.Integrator.SetPositions", [&]() {
                    vbd.SetPositions(X);
                });
            },
            "|#dims|x|#vertices| vertex positions")
        .def_property(
            "v",
            [](Integrator const& vbd) {
                return pbat::profiling::Profile("pbat.gpu.vbd.Integrator.GetVelocities", [&]() {
                    GpuMatrixX v = vbd.GetVelocities();
                    return v;
                });
            },
            [](Integrator& vbd, Eigen::Ref<GpuMatrixX const> const& v) {
                pbat::profiling::Profile("pbat.gpu.vbd.Integrator.SetVelocities", [&]() {
                    vbd.SetVelocities(v);
                });
            },
            "|#dims|x|#vertices| vertex velocities")
        .def_property(
            "a",
            nullptr,
            [](Integrator& vbd, Eigen::Ref<GpuMatrixX const> const& a) {
                pbat::profiling::Profile("pbat.gpu.vbd.Integrator.SetExternalAcceleration", [&]() {
                    vbd.SetExternalAcceleration(a);
                });
            },
            "|#dims|x|#vertices| vertex external accelerations")
        .def_property(
            "m",
            nullptr,
            [](Integrator& vbd, Eigen::Ref<GpuMatrixX const> const& m) {
                pbat::profiling::Profile("pbat.gpu.vbd.Integrator.SetMass", [&]() {
                    vbd.SetMass(m);
                });
            },
            "|#vertices| lumped masses")
        .def_property(
            "wg",
            nullptr,
            [](Integrator& vbd, Eigen::Ref<GpuMatrixX const> const& wg) {
                pbat::profiling::Profile("pbat.gpu.vbd.Integrator.SetQuadratureWeights", [&]() {
                    vbd.SetQuadratureWeights(wg);
                });
            },
            "|#elements| array of quadrature weights")
        .def_property(
            "GNe",
            nullptr,
            [](Integrator& vbd, Eigen::Ref<GpuMatrixX const> const& GP) {
                pbat::profiling::Profile(
                    "pbat.gpu.vbd.Integrator.SetShapeFunctionGradients",
                    [&]() { vbd.SetShapeFunctionGradients(GP); });
            },
            "4x3x|#elements| array of shape function gradients, stored column-wise (i.e. the 4x3 "
            "element shape function gradient matrices are flattened in column-major format)")
        .def_property(
            "lame",
            nullptr,
            [](Integrator& vbd, Eigen::Ref<GpuMatrixX const> const& lame) {
                pbat::profiling::Profile("pbat.gpu.vbd.Integrator.SetLameCoefficients", [&]() {
                    vbd.SetLameCoefficients(lame);
                });
            },
            "2x|#elements| Lame coefficients")
        .def_property(
            "detH_residual",
            nullptr,
            [](Integrator& vbd, GpuScalar zero) {
                vbd.SetNumericalZeroForHessianDeterminant(zero);
            },
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
                pbat::profiling::Profile(
                    "pbat.gpu.vbd.Integrator.SetVertexTetrahedronAdjacencyList",
                    [&]() {
                        vbd.SetVertexTetrahedronAdjacencyList(
                            std::get<0>(GVT),
                            std::get<1>(GVT),
                            std::get<2>(GVT));
                    });
            },
            "3-tuple (prefix,neighbours,data) representing the compressed column storage graph "
            "representation of the vertex-tetrahedron adjacency list. The data property yields the "
            "local vertex index associated with a pair (i,e) of vertex i adjacent to element e.")
        .def_property(
            "kD",
            nullptr,
            [](Integrator& vbd, GpuScalar kD) { vbd.SetRayleighDampingCoefficient(kD); },
            "Sets a uniform Rayleigh damping coefficient on the mesh.")
        .def_property(
            "partitions",
            nullptr,
            [](Integrator& vbd,
               std::pair<Eigen::Ref<GpuIndexVectorX const>, Eigen::Ref<GpuIndexVectorX const>>
                   partitions) { vbd.SetVertexPartitions(partitions.first, partitions.second); },
            "Set vertex partitions for the parallel time integration minimization solve as list of "
            "lists of vertex indices")
        .def_property(
            "strategy",
            nullptr,
            [](Integrator& vbd, EInitializationStrategy strategy) {
                vbd.SetInitializationStrategy(strategy);
            },
            "Set VBD's time step minimization initialization strategy")
        .def(
            "set_gpu_block_size",
            &Integrator::SetBlockSize,
            pyb::arg("num_threads_per_block") = 64,
            "Sets the number of threads per GPU thread block used for time integration "
            "minimization.")
        .def(
            "use_parallel_reduction",
            &Integrator::UseParallelReduction,
            pyb::arg("use") = true,
            "Use parallel reduction to accumulate vertex derivatives.");
#endif // PBAT_USE_CUDA
}

} // namespace vbd
} // namespace gpu
} // namespace py
} // namespace pbat