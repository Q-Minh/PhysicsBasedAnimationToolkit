#ifndef PBAT_MATH_LINALG_MINI_NORM_H
#define PBAT_MATH_LINALG_MINI_NORM_H

#include "Concepts.h"
#include "Reductions.h"
#include "pbat/HostDevice.h"

#include <math.h>
#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto SquaredNorm(TMatrix&& A)
{
    return Dot(std::forward<TMatrix>(A), std::forward<TMatrix>(A));
}

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Norm(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    using ScalarType = typename MatrixType::ScalarType;
    if constexpr (std::is_same_v<ScalarType, float>)
    {
        return sqrtf(SquaredNorm(std::forward<TMatrix>(A)));
    }
    else
    {
        return sqrt(SquaredNorm(std::forward<TMatrix>(A)));
    }
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_NORM_H