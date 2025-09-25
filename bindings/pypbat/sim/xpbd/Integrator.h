/**
 * @file Integrator.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_SIM_XPBD_INTEGRATOR_H
#define PYPBAT_SIM_XPBD_INTEGRATOR_H

#include <nanobind/nanobind.h>

namespace pbat::py::sim::xpbd {

void BindIntegrator(nanobind::module_& m);

} // namespace pbat::py::sim::xpbd

#endif // PYPBAT_SIM_XPBD_INTEGRATOR_H
