#include "Contact.h"

#include "MultibodyMeshMixedCcdDcd.h"

namespace pbat::py::sim::contact {

void Bind(pybind11::module& m)
{
    BindMultibodyMeshMixedCcdDcd(m);
}

} // namespace pbat::py::sim::contact