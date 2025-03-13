#include "Buffer.h"

#include <pbat/gpu/common/Buffer.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace gpu {
namespace common {

void BindBuffer([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_CUDA
    namespace pyb = pybind11;

    using pbat::gpu::common::Buffer;

    pyb::enum_<Buffer::EType>(m, "dtype")
        .value("uint8", Buffer::EType::uint8)
        .value("uint16", Buffer::EType::uint16)
        .value("uint32", Buffer::EType::uint32)
        .value("uint64", Buffer::EType::uint64)
        .value("int8", Buffer::EType::int8)
        .value("int16", Buffer::EType::int16)
        .value("int32", Buffer::EType::int32)
        .value("int64", Buffer::EType::int64)
        .value("float32", Buffer::EType::float32)
        .value("float64", Buffer::EType::float64)
        .export_values();

    pyb::class_<Buffer>(m, "Buffer")
        .def(
            pyb::init<GpuIndex, GpuIndex, Buffer::EType>(),
            pyb::arg("dims") = 1,
            pyb::arg("n"),
            pyb::arg("dtype"))
        .def(pyb::init<Buffer::Data<std::uint8_t> const&>())
        .def(pyb::init<Buffer::Data<std::uint16_t> const&>())
        .def(pyb::init<Buffer::Data<std::uint32_t> const&>())
        .def(pyb::init<Buffer::Data<std::uint64_t> const&>())
        .def(pyb::init<Buffer::Data<std::int8_t> const&>())
        .def(pyb::init<Buffer::Data<std::int16_t> const&>())
        .def(pyb::init<Buffer::Data<std::int32_t> const&>())
        .def(pyb::init<Buffer::Data<std::int64_t> const&>())
        .def(pyb::init<Buffer::Data<float> const&>())
        .def(pyb::init<Buffer::Data<double> const&>())
        .def("set", [](Buffer& b, Buffer::Data<std::uint8_t> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<std::uint16_t> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<std::uint32_t> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<std::uint64_t> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<std::int8_t> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<std::int16_t> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<std::int32_t> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<std::int64_t> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<float> const& data) { b = data; })
        .def("set", [](Buffer& b, Buffer::Data<double> const& data) { b = data; })
        .def("resize", [](Buffer& b, GpuIndex rows, GpuIndex cols) { b.Resize(rows, cols); })
        .def("resize", [](Buffer& b, GpuIndex size) { b.Resize(size); })
        .def_property_readonly("dims", &Buffer::Dims)
        .def_property_readonly("type", &Buffer::Type)
        .def_property_readonly("size", &Buffer::Size);
#endif // PBAT_USE_CUDA
}

} // namespace common
} // namespace gpu
} // namespace py
} // namespace pbat