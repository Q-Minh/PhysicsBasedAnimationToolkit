#ifndef PBAT_MATH_LINALG_MINI_RESHAPE_H
#define PBAT_MATH_LINALG_MINI_RESHAPE_H

#include "Api.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix, int Rows, int Cols, bool RowMajor = false>
class ReshapedView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = ReshapedView<NestedType, Rows, Cols, RowMajor>;

    static_assert(
        Rows * Cols == NestedType::kRows * NestedType::kCols,
        "Reshape dimensions must preserve total number of elements");

    static auto constexpr kRows     = Rows;
    static auto constexpr kCols     = Cols;
    static bool constexpr bRowMajor = RowMajor;

    PBAT_HOST_DEVICE ReshapedView(NestedType const& A) : mA(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        auto const k = bRowMajor ? (i * kCols + j) : (j * kRows + i);
        return mA(k);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const
    {
        return bRowMajor ? mA(i) : mA(i);
    }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    NestedType const& mA;
};

template <int Rows, int Cols, bool RowMajor = false, class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Reshape(TMatrix&& A)
{
    using MatrixType = std::decay_t<TMatrix>;
    return ReshapedView<MatrixType, Rows, Cols, RowMajor>(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_RESHAPE_H
