#include "Profiling.h"

#include <nanobind/stl/string_view.h>
#include <pbat/gpu/profiling/Profiling.h>

namespace pbat::py::gpu::profiling {

void Bind(nanobind::module_& m)
{
    namespace nb = nanobind;
    nb::class_<pbat::gpu::profiling::CudaProfiler>(m, "CudaProfiler")
        .def(
            nb::init<std::string_view>(),
            nb::arg("context"),
            "Create a new GPU Profiler with the given context name")
        .def("start", &pbat::gpu::profiling::CudaProfiler::Start, "Start profiling CUDA API calls")
        .def("stop", &pbat::gpu::profiling::CudaProfiler::Stop, "Stop profiling CUDA API calls");
}

} // namespace pbat::py::gpu::profiling