#include "Device.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <pbat/geometry/Device.h>

namespace pbat::py::geometry {

void BindDevice(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::geometry::Device;
    using pbat::geometry::DeviceConfig;

    nb::class_<DeviceConfig>(m, "DeviceConfig")
        .def(nb::init<>())
        .def_rw("threads", &DeviceConfig::threads, "Number of build threads (0=all, -1=default)")
        .def_rw(
            "user_threads",
            &DeviceConfig::userThreads,
            "Number of user threads used to join scene commits (-1=unspecified)")
        .def_rw(
            "set_affinity",
            &DeviceConfig::setAffinity,
            "Pin threads to cores (0/1, -1=unspecified)")
        .def_rw(
            "start_threads",
            &DeviceConfig::startThreads,
            "Start threads at device creation (0/1, -1=unspecified)")
        .def_rw(
            "isa",
            &DeviceConfig::isa,
            "Instruction set architecture to use (e.g. 'sse2', 'sse4.2', 'avx', 'avx2', 'avx512', "
            "'' = default)")
        .def_rw("max_isa", &DeviceConfig::maxIsa, "Maximum instruction set architecture to use")
        .def_rw("verbose", &DeviceConfig::verbose, "Verbosity level (0..4, -1=unspecified)")
        .def_rw(
            "frequency_level",
            &DeviceConfig::frequencyLevel,
            "Frequency level (e.g. 'simd128', 'simd256', 'simd512', '' = default)")
        .def("to_string", &DeviceConfig::ToString, "Convert config to a human-readable string.");

    nb::class_<Device>(m, "Device")
        .def(
            nb::init<DeviceConfig const&>(),
            nb::arg("config"),
            "Construct a native spatial-acceleration device.\n\n"
            "Args:\n"
            "    config (pbat.geometry.DeviceConfig): Device configuration.\n")
        .def(
            "valid",
            [](Device const& self) { return static_cast<bool>(self); },
            "Check whether the device is valid (has an active native handle).\n\n"
            "Returns:\n"
            "    bool: True if valid, False otherwise.\n");
}

} // namespace pbat::py::geometry
