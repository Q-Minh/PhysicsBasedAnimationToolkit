#include "Profiling.h"

#include <functional>
#include <pbat/profiling/Profiling.h>
#include <string_view>

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
    m.def(
        "profile",
        [](std::function<void()> const& f, std::string_view zoneName) {
            pbat::profiling::Profile(zoneName, f);
        },
        "Profile input function evaluation");
}

} // namespace profiling
} // namespace py
} // namespace pbat