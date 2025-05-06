#ifndef PYPBAT_IO_ARCHIVE_H
#define PYPBAT_IO_ARCHIVE_H

#include <pybind11/pybind11.h>

namespace pbat::py::io {

void BindArchive(pybind11::module& m);

} // namespace pbat::py::io

#endif // PYPBAT_IO_ARCHIVE_H
