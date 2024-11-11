#ifndef PBAT_MATH_LINALG_MINI_GEOMETRY_H
#define PBAT_MATH_LINALG_MINI_GEOMETRY_H

#include "Api.h"
#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
class CrossProduct
{
  public:
    static_assert(
        ((TLhsMatrix::kRows == 3 and TLhsMatrix::kCols == 1) or
         (TLhsMatrix::kRows == 1 and TLhsMatrix::kCols == 3)) and
            ((TRhsMatrix::kRows == 3 and TRhsMatrix::kCols == 1) or
             (TRhsMatrix::kRows == 1 and TRhsMatrix::kCols == 3)),
        "Cross product only valid for 3x1 or 1x3 matrices");

    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using ScalarType = LhsNestedType::ScalarType;
    using SelfType   = CrossProduct<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows     = 3;
    static auto constexpr kCols     = 1;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE CrossProduct(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        j = (i + 1) % 3;
        i = (i + 2) % 3;
        return A(j, 0) * B(i, 0) - A(i, 0) * B(j, 0);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i, 0); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

  private:
    LhsNestedType const& A;
    RhsNestedType const& B;
};

template <class /*CMatrix*/ TLhsMatrix, class /*CMatrix*/ TRhsMatrix>
PBAT_HOST_DEVICE auto Cross(TLhsMatrix&& A, TRhsMatrix&& B)
{
    using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;
    using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;
    return CrossProduct<LhsMatrixType, RhsMatrixType>(
        std::forward<TLhsMatrix>(A),
        std::forward<TRhsMatrix>(B));
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_GEOMETRY_H