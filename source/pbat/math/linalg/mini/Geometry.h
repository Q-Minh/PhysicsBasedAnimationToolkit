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

    PBAT_HOST_DEVICE CrossProduct(LhsNestedType const& _A, RhsNestedType const& _B) : A(_A), B(_B)
    {
    }

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto /*j*/) const
    {
        auto const j = (i + 1) % 3;
        auto const k = (i + 2) % 3;
        return A(j, 0) * B(k, 0) - A(k, 0) * B(j, 0);
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

template <class /*CMatrix*/ TMatrix>
class CrossProductOperator
{
  public:
    static_assert(
        (TMatrix::kRows == 3 and TMatrix::kCols == 1) or
            (TMatrix::kRows == 1 and TMatrix::kCols == 3),
        "Cross product operator only valid for 3x1 or 1x3 matrices");

    using NestedType = TMatrix;
    using ScalarType = typename NestedType::ScalarType;
    using SelfType   = CrossProductOperator<NestedType>;

    static auto constexpr kRows     = 3;
    static auto constexpr kCols     = 3;
    static bool constexpr bRowMajor = false;

    PBAT_HOST_DEVICE CrossProductOperator(NestedType const& _A) : A(_A) {}

    PBAT_HOST_DEVICE ScalarType operator()(auto i, auto j) const
    {
        // Skew-symmetric matrix [a]_x such that [a]_x * b = a x b
        // [  0  -a2   a1 ]
        // [ a2    0  -a0 ]
        // [-a1   a0    0 ]
        // 
        // For off-diagonal elements:
        // (0,1) -> -a2, (0,2) ->  a1
        // (1,0) ->  a2, (1,2) -> -a0
        // (2,0) -> -a1, (2,1) ->  a0
        // For diagonal elements (i==j), k=3-2i which still indexes A, but sign becomes 0.
        auto const k    = 3 - i - j;
        auto const sign = static_cast<ScalarType>((i - j + 3) % 3) - static_cast<ScalarType>((j - i + 3) % 3);
        return sign * A(k, 0);
    }

    PBAT_MINI_READ_API(SelfType)

  private:
    NestedType const& A;
};

template <class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE auto CrossMatrix(TMatrix&& A)
{
    using MatrixType = std::remove_cvref_t<TMatrix>;
    return CrossProductOperator<MatrixType>(std::forward<TMatrix>(A));
}

/**
 * @brief Constructs the skew-symmetric cross product matrix [a]_x in-place.
 * 
 * Given vector a, writes the matrix:
 * [  0  -a2   a1 ]
 * [ a2    0  -a0 ]
 * [-a1   a0    0 ]
 * 
 * @param a Input 3x1 or 1x3 vector
 * @param M Output 3x3 writable matrix
 */
template <class /*CMatrix*/ TVector, class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE void ToSkewSymmetricMatrix(TVector const& a, TMatrix& M)
{
    static_assert(
        (TVector::kRows == 3 and TVector::kCols == 1) or
            (TVector::kRows == 1 and TVector::kCols == 3),
        "Input must be a 3x1 or 1x3 vector");
    static_assert(
        TMatrix::kRows == 3 and TMatrix::kCols == 3,
        "Output must be a 3x3 matrix");

    using ScalarType = typename TVector::ScalarType;

    M(0, 0) = ScalarType{0};
    M(0, 1) = -a(2, 0);
    M(0, 2) = a(1, 0);
    M(1, 0) = a(2, 0);
    M(1, 1) = ScalarType{0};
    M(1, 2) = -a(0, 0);
    M(2, 0) = -a(1, 0);
    M(2, 1) = a(0, 0);
    M(2, 2) = ScalarType{0};
}

/**
 * @brief Constructs the skew-symmetric cross product matrix [a]_x in-place using vectorized access.
 * 
 * Given vector a, writes the matrix (assuming column-major storage):
 * [  0  -a2   a1 ]
 * [ a2    0  -a0 ]
 * [-a1   a0    0 ]
 * 
 * @param a Input 3x1 or 1x3 vector
 * @param M Output 3x3 writable matrix with vectorized (linear) access
 */
template <class /*CMatrix*/ TVector, class /*CMatrix*/ TMatrix>
PBAT_HOST_DEVICE void ToSkewSymmetricMatrixVectorized(TVector const& a, TMatrix& M)
{
    static_assert(
        (TVector::kRows == 3 and TVector::kCols == 1) or
            (TVector::kRows == 1 and TVector::kCols == 3),
        "Input must be a 3x1 or 1x3 vector");
    static_assert(
        TMatrix::kRows == 3 and TMatrix::kCols == 3,
        "Output must be a 3x3 matrix");
    static_assert(
        not TMatrix::bRowMajor,
        "Vectorized access assumes column-major storage");

    using ScalarType = typename TVector::ScalarType;

    // Column-major layout:
    // M[0] = M(0,0), M[1] = M(1,0), M[2] = M(2,0)
    // M[3] = M(0,1), M[4] = M(1,1), M[5] = M(2,1)
    // M[6] = M(0,2), M[7] = M(1,2), M[8] = M(2,2)
    M[0] = ScalarType{0};
    M[1] = a[2];
    M[2] = -a[1];
    M[3] = -a[2];
    M[4] = ScalarType{0};
    M[5] = a[0];
    M[6] = a[1];
    M[7] = -a[0];
    M[8] = ScalarType{0};
}

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_GEOMETRY_H