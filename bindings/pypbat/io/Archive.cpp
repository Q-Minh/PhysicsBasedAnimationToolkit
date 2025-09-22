#include "Archive.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <pbat/io/Archive.h>

namespace pbat::py::io {

void pbat::py::io::BindArchive(nanobind::module_& m)
{
    namespace nb = nanobind;

    nb::enum_<HighFive::File::AccessMode>(m, "AccessMode")
        .value("None", HighFive::File::AccessMode::None)
        .value("ReadOnly", HighFive::File::AccessMode::ReadOnly, "Open file in read-only mode")
        .value("ReadWrite", HighFive::File::AccessMode::ReadWrite, "Open file in read-write mode")
        .value(
            "Truncate",
            HighFive::File::AccessMode::Truncate,
            "Truncate file if it already exists")
        .value("Excl", HighFive::File::AccessMode::Excl, "Fail if file already exists")
        .value("Debug", HighFive::File::AccessMode::Debug, "Open file in debug mode")
        .value("Create", HighFive::File::AccessMode::Create, "Create file if it does not exist")
        .value(
            "Overwrite",
            HighFive::File::AccessMode::Overwrite,
            "Open file in read-write mode, create if it does not exist, and truncate if it does")
        .value(
            "OpenOrCreate",
            HighFive::File::AccessMode::OpenOrCreate,
            "Open file in read-write mode, or create if it does not exist")
        .export_values();

    nb::class_<pbat::io::Archive>(m, "Archive")
        .def(
            nb::init<std::filesystem::path, HighFive::File::AccessMode>(),
            nb::arg("filepath"),
            nb::arg("flags") = HighFive::File::OpenOrCreate)
        .def_prop_ro("usable", &pbat::io::Archive::IsUsable, "Whether the archive is usable")
        .def_prop_ro("path", &pbat::io::Archive::GetPath, "Path of the current HDF5 object")
        .def(
            "unlink",
            &pbat::io::Archive::Unlink,
            nb::arg("path"),
            "Unlink a dataset or group from the archive")
        .def(
            "__getitem__",
            [](pbat::io::Archive& archive, const std::string& path) { return archive[path]; },
            nb::arg("path"),
            "Get a group or create it if it does not exist\n\n"
            "Args:\n"
            "    path (str): Path to the group\n\n"
            "Returns:\n"
            "    Archive: Archive object representing the group\n\n");
}

} // namespace pbat::py::io
