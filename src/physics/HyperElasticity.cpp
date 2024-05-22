#include "pba/physics/HyperElasticity.h"

namespace pba {
namespace physics {

std::pair<Scalar, Scalar> LameCoefficients(Scalar Y, Scalar nu)
{
    Scalar const mu     = Y / (2. * (1. + nu));
    Scalar const lambda = Y * nu / ((1. + nu) * (1. - 2. * nu));
    return {mu, lambda};
}

} // namespace physics
} // namespace pba