#include "Archive.h"

#include <pbat/io/Archive.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace pbat::py::io {

void pbat::py::io::BindArchive(pybind11::module& m)
{
    namespace pyb = pybind11;

    pyb::enum_<HighFive::File::AccessMode>(m, "AccessMode")
        .value("None", HighFive::File::AccessMode::None)
        .value("ReadOnly", HighFive::File::AccessMode::ReadOnly)
        .value("ReadWrite", HighFive::File::AccessMode::ReadWrite)
        .value("Truncate", HighFive::File::AccessMode::Truncate)
        .value("Excl", HighFive::File::AccessMode::Excl)
        .value("Debug", HighFive::File::AccessMode::Debug)
        .value("Create", HighFive::File::AccessMode::Create)
        .value("Overwrite", HighFive::File::AccessMode::Overwrite)
        .value("OpenOrCreate", HighFive::File::AccessMode::OpenOrCreate)
        .export_values();

    pyb::class_<pbat::io::Archive>(m, "Archive")
        .def(
            pyb::init<std::filesystem::path, HighFive::File::AccessMode>(),
            pyb::arg("filepath"),
            pyb::arg("flags") = HighFive::File::OpenOrCreate)
        .def_property_readonly(
            "usable",
            &pbat::io::Archive::IsUsable,
            "Whether the archive is usable")
        .def(
            "__getitem__",
            [](pbat::io::Archive& archive, const std::string& path) { return archive[path]; },
            pyb::arg("path"),
            "Get a group or create it if it does not exist\n\n"
            "Args:\n"
            "    path (str): Path to the group\n\n"
            "Returns:\n"
            "    Archive: Archive object representing the group\n\n");
}

} // namespace pbat::py::io
