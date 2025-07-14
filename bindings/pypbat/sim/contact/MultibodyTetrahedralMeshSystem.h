#ifndef PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H
#define PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H

#include <pybind11/pybind11.h>

namespace pbat::py::sim::contact {

void BindMultibodyTetrahedralMeshSystem(pybind11::module& m);

} // namespace pbat::py::sim::contact

#endif // PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H
