#ifndef PBAT_MATH_LINALG_MINI_FLATTEN_H
#define PBAT_MATH_LINALG_MINI_FLATTEN_H

#include "Api.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TMatrix>
class FlatView
{
  public:
    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = FlatView<NestedType>;

    static auto constexpr kRows     = NestedType::kRows * NestedType::kCols;
    static auto constexpr kCols     = 1;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE FlatView(NestedType const& A) : mA(A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, [[maybe_unused]] auto j) const { return mA(i); }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return mA(i); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return mA(i); }

    PBAT_MINI_READ_WRITE_API(SelfType)

  private:
    NestedType const& mA;
};

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto Flatten(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    return FlatView<MatrixType>(std::forward<TMatrix>(A));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_FLATTEN_H