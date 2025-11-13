/**
 * @file MeshDynamics.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Python bindings for mesh contact dynamics.
 * @version 0.1
 * @date 2025-11-12
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_SIM_CONTACT_MESHDYNAMICS_H
#define PYPBAT_SIM_CONTACT_MESHDYNAMICS_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::contact {

void BindMeshDynamics(nanobind::module_& m);

} // namespace pbat::py::sim::contact

#endif // PYPBAT_SIM_CONTACT_MESHDYNAMICS_H
