#ifndef PBAT_MATH_LINALG_MINI_PRODUCT_H
#define PBAT_MATH_LINALG_MINI_PRODUCT_H

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
class Product
{
  public:
    using LhsNestedType = TLhsMatrix;
    using RhsNestedType = TRhsMatrix;

    using Scalar   = typename LhsNestedType::Scalar;
    using SelfType = Product<LhsNestedType, RhsNestedType>;

    static auto constexpr RowsAtCompileTime = LhsNestedType::RowsAtCompileTime;
    static auto constexpr ColsAtCompileTime = RhsNestedType::ColsAtCompileTime;

    PBAT_HOST_DEVICE Product(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B) {}

    PBAT_HOST_DEVICE constexpr auto Rows() const { return RowsAtCompileTime; }
    PBAT_HOST_DEVICE constexpr auto Cols() const { return ColsAtCompileTime; }

    PBAT_HOST_DEVICE auto operator()(auto i, auto j) const
    {
        auto contract = [this, i, j]<auto... K>(std::index_sequence<K...>) {
            return ((A(i, K) * B(K, j)) + ...);
        };
        return contract(std::make_index_sequence<LhsNestedType::ColsAtCompileTime>());
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE auto operator()(auto i) const
    {
        return (*this)(i % RowsAtCompileTime, i / RowsAtCompileTime);
    }

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

template <CMatrix TLhsMatrix, CMatrix TRhsMatrix>
PBAT_HOST_DEVICE auto operator*(TLhsMatrix const& A, TRhsMatrix const& B)
{
    return Product<TLhsMatrix, TRhsMatrix>(A, B);
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_PRODUCT_H