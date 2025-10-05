/**
 * @file MultibodyTetrahedralMeshSystem.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H
#define PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::contact {

void BindMultibodyTetrahedralMeshSystem(nanobind::module_& m);

} // namespace pbat::py::sim::contact

#endif // PYPBAT_SIM_CONTACT_MULTIBODYTETRAHEDRALMESH_H
