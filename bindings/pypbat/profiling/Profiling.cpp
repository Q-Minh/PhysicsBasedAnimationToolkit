#include "Profiling.h"

#include <functional>
#include <pbat/Aliases.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <string>
#include <tuple>

namespace pbat {
namespace py {
namespace profiling {

// template <class... Args>
// inline void BindProfile(pybind11::module& m)
// {
//     using TupleType = std::tuple<Args...>;
//     m.def(
//         "profile",
//         [](std::string const& zoneName, std::function<TupleType()> const& f) -> TupleType {
//             return pbat::profiling::Profile(zoneName, f);
//         },
//         "Profile input function evaluation");
// }

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
        [](std::string const& zoneName, std::function<void()> const& f) {
            pbat::profiling::Profile(zoneName, f);
        },
        "Profile input function evaluation");

    // // As a syntactic sugar thing for the Python interface, generate bindings for probable return
    // // types. Otherwise, just use the void *f() overload of profile, but might not be super
    // // 'pythonic'.

    // // These return types typically result from linear solves, or dense matrix decompositions.
    // BindProfile<MatrixX>(m);
    // BindProfile<MatrixX, MatrixX>(m);
    // BindProfile<MatrixX, MatrixX, MatrixX>(m);
    // BindProfile<MatrixX, MatrixX, MatrixX, MatrixX>(m);

    // // These return types might result from graph partitionings, or graph constructions
    // BindProfile<IndexVectorX>(m);
    // BindProfile<IndexMatrixX>(m);
    // BindProfile<CSCMatrix>(m);
    // BindProfile<CSRMatrix>(m);
}

} // namespace profiling
} // namespace py
} // namespace pbat