#include "Io.h"

#include "Archive.h"

namespace pbat::py::io {

void Bind(pybind11::module& m)
{
    BindArchive(m);
}

} // namespace pbat::py::io