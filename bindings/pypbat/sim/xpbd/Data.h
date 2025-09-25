/**
 * @file Data.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-25
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef PYPBAT_SIM_XPBD_DATA_H
#define PYPBAT_SIM_XPBD_DATA_H

#include <nanobind/nanobind.h>

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void BindData(nanobind::module_& m);

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat

#endif // PYPBAT_SIM_XPBD_DATA_H
