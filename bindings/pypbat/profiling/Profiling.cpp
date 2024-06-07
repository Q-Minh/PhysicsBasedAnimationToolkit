#include "Profiling.h"

#include "pbat/profiling/Profiling.h"

namespace pbat {
namespace py {
namespace profiling {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
    m.def(
        "begin_frame",
        &pbat::profiling::BeginFrame,
        "Start new profiling frame",
        pyb::arg("name"));
    m.def("end_frame", &pbat::profiling::EndFrame, "End current profiling frame", pyb::arg("name"));
    m.def(
        "is_connected_to_server",
        &pbat::profiling::IsConnectedToServer,
        "Check if profiler has connected to profiling server");
}

} // namespace profiling
} // namespace py
} // namespace pbat