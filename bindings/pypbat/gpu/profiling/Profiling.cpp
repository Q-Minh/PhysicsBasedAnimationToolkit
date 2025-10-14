#include "Profiling.h"

#include <pbat/gpu/profiling/Profiling.h>

namespace pbat::py::gpu::profiling {

void Bind([[maybe_unused]] nanobind::module_& m)
{
#ifdef PBAT_USE_CUDA
    namespace nb = nanobind;
    nb::class_<pbat::gpu::profiling::CudaProfiler>(m, "CudaProfiler")
        .def(nb::init<>())
        .def("start", &pbat::gpu::profiling::CudaProfiler::Start, "Start profiling CUDA API calls")
        .def("stop", &pbat::gpu::profiling::CudaProfiler::Stop, "Stop profiling CUDA API calls");
#endif // PBAT_USE_CUDA
}

} // namespace pbat::py::gpu::profiling