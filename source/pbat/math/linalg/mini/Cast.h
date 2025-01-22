#ifndef PBAT_MATH_LINALG_MINI_CAST_H
#define PBAT_MATH_LINALG_MINI_CAST_H

#include "Api.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix, class NewType>
class CastView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = NewType;
    using SelfType   = CastView<NestedType, NewType>;

    static auto constexpr kRows     = NestedType::kRows;
    static auto constexpr kCols     = NestedType::kCols;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE CastView(NestedType const& A) : A(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        return static_cast<ScalarType>(A(i, j));
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    NestedType const& A;
};

template <class NewType, class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Cast(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    return CastView<MatrixType, NewType>(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_CAST_H
