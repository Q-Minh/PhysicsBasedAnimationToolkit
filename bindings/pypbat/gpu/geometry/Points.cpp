#include "Points.h"

#include <pbat/gpu/geometry/Primitives.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void BindPoints([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;
    using namespace pbat::gpu;
    using namespace pbat::gpu::geometry;
    pyb::class_<Points>(m, "Points")
        .def(
            pyb::init([](Eigen::Ref<GpuMatrixX const> const& V) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Points.Construct", [&]() {
                    Points P(V);
                    return P;
                });
            }),
            pyb::arg("V"),
            "Construct points on GPU via |#dims|x|#vertices| array of vertex positions")
        .def_property(
            "V",
            [](Points const& P) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Points.Get", [&]() {
                    GpuMatrixX V = P.Get();
                    return V;
                });
            },
            [](Points& P, Eigen::Ref<GpuMatrixX const> const& V) {
                return pbat::profiling::Profile("pbat.gpu.geometry.Points.Update", [&]() {
                    P.Update(V);
                });
            },
            "Get/set points on GPU via |#dims|x|#vertices| array of vertex positions");
#endif // PBAT_USE_CUDA
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat