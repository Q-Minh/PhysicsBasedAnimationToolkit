#include "Simplices.h"

#include <pbat/gpu/geometry/Primitives.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindSimplices([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;

    using namespace pbat::gpu;
    using namespace pbat::gpu::geometry;
    using ESimplexType = Simplices::ESimplexType;
    pyb::enum_<ESimplexType>(m, "SimplexType")
        .value("Vertex", ESimplexType::Vertex)
        .value("Edge", ESimplexType::Edge)
        .value("Triangle", ESimplexType::Triangle)
        .value("Tetrahedron", ESimplexType::Tetrahedron)
        .export_values();

    pyb::class_<Simplices>(m, "Simplices")
        .def(
            pyb::init([](Eigen::Ref<GpuIndexMatrixX const> const& C) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Simplices.Construct", [&]() {
                    Simplices S(C);
                    return S;
                });
            }),
            pyb::arg("C"),
            "Construct simplices on GPU via |#simplex vertices|x|#simplices| array of simplex "
            "vertex indices")
        .def_property_readonly(
            "type",
            &Simplices::Type,
            "Type of simplex stored by this Simplices instance")
        .def_property_readonly(
            "C",
            [](Simplices const& S) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Simplices.Get", [&]() {
                    GpuIndexMatrixX C = S.Get();
                    return C;
                });
            },
            "Load GPU simplices to CPU");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat