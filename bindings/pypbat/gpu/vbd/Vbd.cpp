#include "Vbd.h"

#include <pbat/gpu/Aliases.h>
#include <pbat/gpu/vbd/InitializationStrategy.h>
#include <pbat/gpu/vbd/Vbd.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <utility>

namespace pbat {
namespace py {
namespace gpu {
namespace vbd {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
#ifdef PBAT_USE_CUDA

    using namespace pbat;
    using pbat::gpu::vbd::EInitializationStrategy;
    using pbat::gpu::vbd::Vbd;

    pyb::enum_<EInitializationStrategy>(m, "InitializationStrategy")
        .value("Position", EInitializationStrategy::Position)
        .value("Inertia", EInitializationStrategy::Inertia)
        .value("KineticEnergyMinimum", EInitializationStrategy::KineticEnergyMinimum)
        .value("AdaptiveVbd", EInitializationStrategy::AdaptiveVbd)
        .value("AdaptivePbat", EInitializationStrategy::AdaptivePbat)
        .export_values();

    pyb::class_<Vbd>(m, "Vbd")
        .def(
            pyb::init([](Eigen::Ref<GpuMatrixX const> const& X,
                         Eigen::Ref<GpuIndexMatrixX const> const& V,
                         Eigen::Ref<GpuIndexMatrixX const> const& F,
                         Eigen::Ref<GpuIndexMatrixX const> const& T) {
                return pbat::profiling::Profile("pbat.gpu.vbd.Vbd.Construct", [&]() {
                    Vbd vbd(X, V, F, T);
                    return vbd;
                });
            }),
            pyb::arg("X"),
            pyb::arg("V"),
            pyb::arg("F"),
            pyb::arg("T"),
            "Construct a VBD algorithm to run on the GPU using input vertex positions X as an "
            "array of dimensions 3x|#vertices|, collision vertices V as an array of dimensions "
            "1x|#collision vertices|, collision triangles F as an array of dimensions "
            "3x|#collision triangles| and tetrahedra T as an array of dimensions 4x|#tetrahedra|.")
        .def(
            "step",
            [](Vbd& vbd, GpuScalar dt, GpuIndex iterations, GpuIndex substeps, GpuScalar rho) {
                pbat::profiling::Profile("pbat.gpu.vbd.Vbd.Step", [&]() {
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
            [](Vbd const& vbd) {
                return pbat::profiling::Profile("pbat.gpu.vbd.Vbd.GetPositions", [&]() {
                    GpuMatrixX X = vbd.GetPositions();
                    return X;
                });
            },
            [](Vbd& vbd, Eigen::Ref<GpuMatrixX const> const& X) {
                pbat::profiling::Profile("pbat.gpu.vbd.Vbd.SetPositions", [&]() {
                    vbd.SetPositions(X);
                });
            },
            "|#dims|x|#vertices| vertex positions")
        .def_property(
            "v",
            [](Vbd const& vbd) {
                return pbat::profiling::Profile("pbat.gpu.vbd.Vbd.GetVelocities", [&]() {
                    GpuMatrixX v = vbd.GetVelocities();
                    return v;
                });
            },
            [](Vbd& vbd, Eigen::Ref<GpuMatrixX const> const& v) {
                pbat::profiling::Profile("pbat.gpu.vbd.Vbd.SetVelocities", [&]() {
                    vbd.SetVelocities(v);
                });
            },
            "|#dims|x|#vertices| vertex velocities")
        .def_property(
            "a",
            nullptr,
            [](Vbd& vbd, Eigen::Ref<GpuMatrixX const> const& a) {
                pbat::profiling::Profile("pbat.gpu.vbd.Vbd.SetExternalAcceleration", [&]() {
                    vbd.SetExternalAcceleration(a);
                });
            },
            "|#dims|x|#vertices| vertex external accelerations")
        .def_property(
            "m",
            nullptr,
            [](Vbd& vbd, Eigen::Ref<GpuMatrixX const> const& m) {
                pbat::profiling::Profile("pbat.gpu.vbd.Vbd.SetMass", [&]() { vbd.SetMass(m); });
            },
            "|#vertices| lumped masses")
        .def_property(
            "wg",
            nullptr,
            [](Vbd& vbd, Eigen::Ref<GpuMatrixX const> const& wg) {
                pbat::profiling::Profile("pbat.gpu.vbd.Vbd.SetQuadratureWeights", [&]() {
                    vbd.SetQuadratureWeights(wg);
                });
            },
            "|#elements| array of quadrature weights")
        .def_property(
            "GNe",
            nullptr,
            [](Vbd& vbd, Eigen::Ref<GpuMatrixX const> const& GP) {
                pbat::profiling::Profile("pbat.gpu.vbd.Vbd.SetShapeFunctionGradients", [&]() {
                    vbd.SetShapeFunctionGradients(GP);
                });
            },
            "4x3x|#elements| array of shape function gradients, stored column-wise (i.e. the 4x3 "
            "element shape function gradient matrices are flattened in column-major format)")
        .def_property(
            "lame",
            nullptr,
            [](Vbd& vbd, Eigen::Ref<GpuMatrixX const> const& lame) {
                pbat::profiling::Profile("pbat.gpu.vbd.Vbd.SetLameCoefficients", [&]() {
                    vbd.SetLameCoefficients(lame);
                });
            },
            "2x|#elements| Lame coefficients")
        .def_property(
            "RdetH",
            nullptr,
            [](Vbd& vbd, GpuScalar zero) { vbd.SetNumericalZeroForHessianDeterminant(zero); },
            "Numerical zero used in Hessian determinant check for approximate singularity "
            "detection")
        .def_property(
            "GVT",
            nullptr,
            [](Vbd& vbd,
               std::tuple<
                   Eigen::Ref<GpuIndexVectorX const>,
                   Eigen::Ref<GpuIndexVectorX const>,
                   Eigen::Ref<GpuIndexVectorX const>> const& GVT) {
                pbat::profiling::Profile(
                    "pbat.gpu.vbd.Vbd.SetVertexTetrahedronAdjacencyList",
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
            [](Vbd& vbd, GpuScalar kD) { vbd.SetRayleighDampingCoefficient(kD); },
            "Sets a uniform Rayleigh damping coefficient on the mesh.")
        .def_property(
            "partitions",
            nullptr,
            [](Vbd& vbd, std::vector<std::vector<GpuIndex>> const& partitions) {
                vbd.SetVertexPartitions(partitions);
            },
            "Set vertex partitions for the parallel time integration minimization solve as list of "
            "lists of vertex indices")
        .def_property(
            "initialization_strategy",
            nullptr,
            [](Vbd& vbd, EInitializationStrategy strategy) {
                vbd.SetInitializationStrategy(strategy);
            },
            "Set VBD's time step minimization initialization strategy")
        .def(
            "set_gpu_block_size",
            &Vbd::SetBlockSize,
            pyb::arg("num_threads_per_block") = 64,
            "Sets the number of threads per GPU thread block used for time integration "
            "minimization.");
#endif // PBAT_USE_CUDA
}

} // namespace vbd
} // namespace gpu
} // namespace py
} // namespace pbat