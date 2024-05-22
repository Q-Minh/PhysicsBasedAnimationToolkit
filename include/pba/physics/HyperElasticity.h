#ifndef PBA_PHYSICS_HYPER_ELASTICITY_H
#define PBA_PHYSICS_HYPER_ELASTICITY_H

#include "pba/aliases.h"

namespace pba {
namespace physics {

std::pair<Scalar, Scalar> LameCoefficients(Scalar Y, Scalar nu);

} // namespace physics
} // namespace pba

#endif // PBA_PHYSICS_HYPER_ELASTICITY_H