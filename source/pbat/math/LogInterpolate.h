#ifndef PBAT_MATH_LOG_INTERPOLATE_H
#define PBAT_MATH_LOG_INTERPOLATE_H

#include "pbat/HostDevice.h"

#include <cmath>

namespace pbat::math {

/**
 * @brief Logarithmic interpolation between two values
 *
 * Computes a value that interpolates between `fa` and `fb` logarithmically
 * based on the position of `t` between `a` and `b`.
 *
 * The formula is:
 *   result = fa + (fb - fa) * log(t / a) / log(b / a)
 *
 * @tparam TScalar Scalar type (e.g., float, double)
 * @param fa Value at parameter `a`
 * @param fb Value at parameter `b`
 * @param a Start of parameter range
 * @param b End of parameter range
 * @param t Parameter value to interpolate at (should be in [a, b])
 * @return Logarithmically interpolated value between `fa` and `fb`
 */
template <class TScalar>
PBAT_HOST_DEVICE TScalar LogInterpolate(TScalar fa, TScalar fb, TScalar a, TScalar b, TScalar t)
{
    using namespace std;
    return fa + (fb - fa) * log(t / a) / log(b / a);
}

} // namespace pbat::math

#endif // PBAT_MATH_LOG_INTERPOLATE_H
