#include "Io.h"

#include "Archive.h"

namespace pbat::py::io {

void Bind(nanobind::module_& m)
{
    BindArchive(m);
}

} // namespace pbat::py::io