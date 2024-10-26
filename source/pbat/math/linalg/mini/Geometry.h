#ifndef PBAT_MATH_LINALG_MINI_GEOMETRY_H
#define PBAT_MATH_LINALG_MINI_GEOMETRY_H

#include "Concepts.h"
#include "SubMatrix.h"
#include "Transpose.h"
#include "pbat/HostDevice.h"

#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
class CrossProduct
{
  public:
    static_assert(
        ((TLhsMatrix::RowsAtCompileTime == 3 and TLhsMatrix::ColsAtCompileTime == 1) or
         (TLhsMatrix::RowsAtCompileTime == 1 and TLhsMatrix::ColsAtCompileTime == 3)) and
            ((TRhsMatrix::RowsAtCompileTime == 3 and TRhsMatrix::ColsAtCompileTime == 1) or
             (TRhsMatrix::RowsAtCompileTime == 1 and TRhsMatrix::ColsAtCompileTime == 3)),
        "Cross product only valid for 3x1 or 1x3 matrices");

    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using Scalar   = LhsNestedType::Scalar;
    using SelfType = CrossProduct<LhsNestedType, RhsNestedType>;

    static auto constexpr RowsAtCompileTime = 3;
    static auto constexpr ColsAtCompileTime = 1;

    PBAT_HOST_DEVICE CrossProduct(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const
    {
        j = (i + 1) % 3;
        i = (i + 2) % 3;
        return A(j, 0) * B(i, 0) - A(i, 0) * B(j, 0);
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const { return (*this)(i, 0); }

    template <auto S, auto T>
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, S, T> Slice(auto i, auto j) const
    {
        return ConstSubMatrix<SelfType, S, T>(*this, i, j);
    }
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, RowsAtCompileTime, 1> Col(auto j) const
    {
        return Slice<RowsAtCompileTime, 1>(0, j);
    }
    PBAT_HOST_DEVICE ConstSubMatrix<SelfType, 1, ColsAtCompileTime> Row(auto i) const
    {
        return Slice<1, ColsAtCompileTime>(i, 0);
    }
    PBAT_HOST_DEVICE ConstTransposeView<SelfType> Transpose() const
    {
        return ConstTransposeView<SelfType>(*this);
    }

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