#ifndef PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H
#define PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H

#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/math/linalg/mini/Norm.h"
#include "pbat/math/linalg/mini/Product.h"

#include <Eigen/Core>

namespace pbat::math::optimization {

/**
 * @brief Parameters for trust-region SR1 optimization in a triangle
 * @tparam TScalar Scalar type
 */
template <common::CFloatingPoint TScalar>
struct MinimizeInTriangleWithTrustRegionSr1Params
{
    TScalar R0;   ///< Initial trust region radius
    TScalar eta;  ///< Trust region minimal energy reduction ratio
    TScalar trlo; ///< Largest energy reduction ratio under which to shrink trust region. Must
                  ///< satisfy `0 < trlo < trhi`
    TScalar trhi; ///< Smallest energy reduction ratio over which to grow trust region. Must satisfy
                  ///< `trlo < trhi < 1`
    TScalar trbound;  ///< Smallest step size multiple of trust region radius over which to grow
                      ///< trust region. Must satisfy `0 < trbound <= 1`.
    TScalar trgrow;   ///< Trust region growth factor
    TScalar trshrink; ///< Trust region shrink factor
    TScalar
        delta0; ///< Numerical offset preventing division by zero in cases of zero energy reduction.
    int nMaxIters; ///< Maximum number of solver iterations
};

template <
    class FObjective,
    class FGradient,
    math::linalg::mini::CMatrix TMatrixXk,
    class TScalar = typename TMatrixXk::ScalarType>
bool MinimizeInTriangleWithTrustRegionSr1(
    FObjective const& f,
    FGradient const& gradf,
    TMatrixXk& xk,
    MinimizeInTriangleWithTrustRegionSr1Params<TScalar> const& params)
{
}

} // namespace pbat::math::optimization

#endif // PBAT_MATH_OPTIMIZATION_TRIANGLECONSTRAINEDTRUSTREGIONSR1_H
