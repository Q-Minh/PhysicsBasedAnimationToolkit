/**
 * @file State.h
 * @author your name (you@domain.com)
 * @brief Python bindings for State class in Offset Geometry Contact (OGC) algorithm.
 * @version 0.1
 * @date 2025-12-10
 * @copyright Copyright (c) 2025
 */

#ifndef PYPBAT_SIM_CONTACT_OGC_STATE_H
#define PYPBAT_SIM_CONTACT_OGC_STATE_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::contact::ogc {

void BindState(nanobind::module_& m);

} // namespace pbat::py::sim::contact::ogc

#endif // PYPBAT_SIM_CONTACT_OGC_STATE_H
