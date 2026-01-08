#ifndef PBAT_MATH_LINALG_MINI_NORM_H
#define PBAT_MATH_LINALG_MINI_NORM_H

#include "Concepts.h"
#include "Reductions.h"
#include "pbat/HostDevice.h"

#include <cmath>
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
    using namespace std;
    using MatrixType = std::remove_cvref_t<TMatrix>;
    PBAT_MINI_CHECK_CMATRIX(MatrixType);
    return sqrt(SquaredNorm(std::forward<TMatrix>(A)));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_NORM_H
