#ifndef PBAT_SIM_XPBD_ENUMS_H
#define PBAT_SIM_XPBD_ENUMS_H

namespace pbat {
namespace sim {
namespace xpbd {

enum class EConstraint : int { StableNeoHookean = 0, Collision, NumberOfConstraintTypes };

} // namespace xpbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_XPBD_ENUMS_H
