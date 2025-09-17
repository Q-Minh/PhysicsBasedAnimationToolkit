#include "Contact.h"

#include "VertexTriangleMixedCcdDcd.h"

namespace pbat::py::gpu::contact {

void Bind(nanobind::module_& m)
{
    BindVertexTriangleMixedCcdDcd(m);
}

} // namespace pbat::py::gpu::contact