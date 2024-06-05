#include "Profiling.h"

#include "pbat/profiling/Profiling.h"

namespace pypbat {

void bind_profiling(py::module& m)
{
    m.def(
        "begin_frame",
        &pbat::profiling::BeginFrame,
        "Start new profiling frame",
        py::arg("name"));
    m.def("end_frame", &pbat::profiling::EndFrame, "End current profiling frame", py::arg("name"));
    m.def(
        "is_connected_to_server",
        &pbat::profiling::IsConnectedToServer,
        "Check if profiler has connected to profiling server");
}

} // namespace pypbat