#include "Contact.h"

#include "MeshVertexTetrahedronDcd.h"
#include "MultibodyMeshMixedCcdDcd.h"
#include "MultibodyTetrahedralMeshSystem.h"

namespace pbat::py::sim::contact {

void Bind(pybind11::module& m)
{
    BindMeshVertexTetrahedronDcd(m);
    BindMultibodyMeshMixedCcdDcd(m);
    BindMultibodyTetrahedralMeshSystem(m);
}

} // namespace pbat::py::sim::contact