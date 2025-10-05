#ifndef PBAT_MATH_LINALG_MINI_REPEAT_H
#define PBAT_MATH_LINALG_MINI_REPEAT_H

#include "Api.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix, int RepeatRows, int RepeatCols>
class TiledView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = TiledView<NestedType, RepeatRows, RepeatCols>;

    static auto constexpr kRows     = RepeatRows * NestedType::kRows;
    static auto constexpr kCols     = RepeatCols * NestedType::kCols;
    static bool constexpr bRowMajor = NestedType::bRowMajor;

    PBAT_HOST_DEVICE TiledView(NestedType const& _A) : A(_A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        return A(i % NestedType::kRows, j % NestedType::kCols);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    NestedType const& A;
};

template <auto RepeatRows, auto RepeatCols, class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Repeat(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    return TiledView<MatrixType, RepeatRows, RepeatCols>(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_REPEAT_H