#include "Contact.h"

#include "MultiMesh.h"
#include "MultibodyMeshMixedCcdDcd.h"
#include "OffsetGeometryContact.h"

namespace pbat::py::sim::contact {

void Bind(nanobind::module_& m)
{
    BindMultibodyMeshMixedCcdDcd(m);
    BindOffsetGeometryContact(m);
    BindMultiMesh(m);
}

} // namespace pbat::py::sim::contact