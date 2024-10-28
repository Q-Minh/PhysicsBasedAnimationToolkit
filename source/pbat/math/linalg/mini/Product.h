#ifndef PBAT_MATH_LINALG_MINI_PRODUCT_H
#define PBAT_MATH_LINALG_MINI_PRODUCT_H

#include "Api.h"
#include "Concepts.h"
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

    using ScalarType = typename LhsNestedType::ScalarType;
    using SelfType   = Product<LhsNestedType, RhsNestedType>;

    static auto constexpr kRows     = LhsNestedType::kRows;
    static auto constexpr kCols     = RhsNestedType::kCols;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE Product(LhsNestedType const& A, RhsNestedType const& B) : A(A), B(B) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        auto contract = [this, i, j]<auto... K>(std::index_sequence<K...>) {
            return ((A(i, K) * B(K, j)) + ...);
        };
        return contract(std::make_index_sequence<LhsNestedType::kCols>());
    }

    // Vector(ized) access
    PBAT_HOST_DEVICE ScalarType operator()(auto i) const { return (*this)(i % kRows, i / kRows); }
    PBAT_HOST_DEVICE ScalarType operator[](auto i) const { return (*this)(i); }

    PBAT_MINI_READ_API(SelfType)

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