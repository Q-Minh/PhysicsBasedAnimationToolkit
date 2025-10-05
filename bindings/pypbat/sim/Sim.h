/**
 * @file Sim.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_SIM_SIM_H
#define PYPBAT_SIM_SIM_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace sim {

void Bind(nanobind::module_& m);

} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_SIM_H
