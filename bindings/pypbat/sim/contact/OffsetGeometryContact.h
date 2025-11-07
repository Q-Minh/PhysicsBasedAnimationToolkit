/**
 * @file OffsetGeometryContact.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Python bindings for offset geometry contact algorithm.
 * @version 0.1
 * @date 2025-11-06
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H
#define PYPBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::contact {

void BindOffsetGeometryContact(nanobind::module_& m);

} // namespace pbat::py::sim::contact

#endif // PYPBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H
