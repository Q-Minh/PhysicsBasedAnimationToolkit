import sympy as sp
from sympy.integrals import quadrature


def shifted_gauss_quadrature(order: int, dims: int = 1, ndigits=10):
    x, w = quadrature.gauss_legendre(order, ndigits)
    a, b = 0, 1
    # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
    x = [(b-a)/2 * xi + (a+b)/2 for xi in x]
    w = [(b-a)/2*wi for wi in w]
    if dims == 1:
        x = [[xi] for xi in x]
    if dims == 2:
        x = [[xi, xj] for xi in x for xj in x]
        w = [wi*wj for wi in w for wj in w]
    if dims == 3:
        x = [[xi, xj, xk]
             for xi in x for xj in x for xk in x]
        w = [wi*wj*wk for wi in w for wj in w for wk in w]
    return (x, w)


if __name__ == "__main__":
    header = """
#ifndef PBA_CORE_MATH_GAUSS_QUADRATURE_H
#define PBA_CORE_MATH_GAUSS_QUADRATURE_H

#include "pba/aliases.h"
#include <array>
#include <cstdint>

namespace pba {
namespace math {

/**
 * @brief Shifted Gauss Legendre quadrature scheme over the unit box [0,1] in Dims dimensions.
 *
 * @tparam Dims 
 * @tparam Order 
 */
template <int Dims, int Order>
struct GaussLegendreQuadrature;
    """

    footer = """
} // math
} // pba
    
#endif // PBA_CORE_MATH_GAUSS_QUADRATURE_H
    """

    template = """
template <>
struct GaussLegendreQuadrature<{0},{1}>
{{
    inline static std::uint8_t constexpr kDims = {0};
    inline static std::uint8_t constexpr kOrder = {1};
    inline static int constexpr kPoints = {2};
    inline static std::array<Scalar, (kDims+1)*kPoints> constexpr points = {{{3}}};
    inline static std::array<Scalar, kPoints> constexpr weights = {{{4}}};
}};
    """

    with open("GaussQuadrature.h", "w") as file:
        file.write(header)
        for d in [1, 2, 3]:
            for order in range(1, 11):
                x, w = shifted_gauss_quadrature(order, d)
                for xi in x:
                    xi.insert(0, 1 - sum(xi))
                points = ",".join(
                    [str(xi[j]) for j in range(d+1) for xi in x])
                weights = ",".join([str(wi) for wi in w])
                file.write(template.format(
                    d, order, len(x), points, weights))
        file.write(footer)
