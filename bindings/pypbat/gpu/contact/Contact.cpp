#include "Contact.h"

#include "VertexTriangleMixedCcdDcd.h"

namespace pbat::py::gpu::contact {

void Bind(pybind11::module& m)
{
    BindVertexTriangleMixedCcdDcd(m);
}

} // namespace pbat::py::gpu::contact