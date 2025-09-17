#ifndef PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H
#define PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::contact {

void BindMultibodyTetrahedralMeshSystem(nanobind::module_& m);

} // namespace pbat::py::sim::contact

#endif // PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H
