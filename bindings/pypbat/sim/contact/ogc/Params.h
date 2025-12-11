/**
 * @file Params.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Python bindings for Params class in Offset Geometry Contact (OGC) algorithm.
 * @version 0.1
 * @date 2025-12-10
 * @copyright Copyright (c) 2025
 */

#ifndef PYPBAT_SIM_CONTACT_OGC_PARAMS_H
#define PYPBAT_SIM_CONTACT_OGC_PARAMS_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::contact::ogc {

void BindParams(nanobind::module_& m);

} // namespace pbat::py::sim::contact::ogc

#endif // PYPBAT_SIM_CONTACT_OGC_PARAMS_H
