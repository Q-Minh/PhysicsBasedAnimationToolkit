#include "Contact.h"

#include "MeshDynamics.h"
#include "MeshSdfContact.h"
#include "MultiMesh.h"
#include "MultibodyMeshMixedCcdDcd.h"
#include "ogc/Ogc.h"

namespace pbat::py::sim::contact {

void Bind(nanobind::module_& m)
{
    auto mogc = m.def_submodule("ogc", "Offset Geometry Contact (OGC) algorithm bindings");
    BindMultiMesh(m);
    ogc::Bind(mogc);
    BindMultibodyMeshMixedCcdDcd(m);
    BindMeshSdfContact(m);
    BindMeshDynamics(m);
}

} // namespace pbat::py::sim::contact