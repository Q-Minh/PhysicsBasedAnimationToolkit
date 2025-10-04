#ifndef PBAT_SIM_CONTACT_CONSTRAINTS_H
#define PBAT_SIM_CONTACT_CONSTRAINTS_H

#include "pbat/HostDevice.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/Matrix.h"

namespace pbat::sim::contact {
/**
 * @brief Computes the constraint \f$ C(x) = \begin{bmatrix} N & T & B \end{bmatrix}^T
 * (\begin{bmatrix} x_i & x_j & x_k \end{bmatrix} \beta - P) \in \mathbb{R}^3 \f$
 *
 * where \f$ N, T, B \f$ are an orthonormal contact basis, \f$ P \f$ is its origin, \f$ x_i, x_j,
 * x_k \f$ are triangle vertex positions in counter-clockwise order, and \f$ \beta \f$ are
 * barycentric coordinates of the contact point on the triangle.
 *
 * @tparam TMatrixP Matrix type for \f$ P \f$
 * @tparam TMatrixNTB Matrix type for \f$ \begin{bmatrix} N & T & B \end{bmatrix} \f$
 * @tparam TMatrixX1 Matrix type for \f$ x_i \f$
 * @tparam TMatrixX2 Matrix type for \f$ x_j \f$
 * @tparam TMatrixX3 Matrix type for \f$ x_k \f$
 * @tparam TMatrixBary Matrix type for \f$ bary \f$
 * @tparam TMatrixC Matrix type for \f$ C \f$
 * @param P `3 x 1` origin of the contact basis
 * @param NTB `3 x 3` contact basis (normal, tangent, bitangent)
 * @param X1 `3 x 1` vertex position \f$ x_i \f$
 * @param X2 `3 x 1` vertex position \f$ x_j \f$
 * @param X3 `3 x 1` vertex position \f$ x_k \f$
 * @param bary `3 x 1` barycentric coordinates of the contact point on the triangle
 * @param C `3 x 1` constraint (vector) value
 */
template <
    math::linalg::mini::CMatrix TMatrixP,
    math::linalg::mini::CMatrix TMatrixNTB,
    math::linalg::mini::CMatrix TMatrixX1,
    math::linalg::mini::CMatrix TMatrixX2,
    math::linalg::mini::CMatrix TMatrixX3,
    math::linalg::mini::CMatrix TMatrixBary,
    math::linalg::mini::CMatrix TMatrixC>
PBAT_HOST_DEVICE void TriangleEnvironmentConstraint(
    TMatrixP const& P,
    TMatrixNTB const& NTB,
    TMatrixX1 const& X1,
    TMatrixX2 const& X2,
    TMatrixX3 const& X3,
    TMatrixBary const& bary,
    TMatrixC& C)
{
    static_assert(TMatrixP::kRows == 3 and TMatrixP::kCols == 1, "P must be 3x1");
    static_assert(TMatrixNTB::kRows == 3 and TMatrixNTB::kCols == 3, "NTB must be 3x3");
    static_assert(TMatrixX1::kRows == 3 and TMatrixX1::kCols == 1, "X1 must be 3x1");
    static_assert(TMatrixX2::kRows == 3 and TMatrixX2::kCols == 1, "X2 must be 3x1");
    static_assert(TMatrixX3::kRows == 3 and TMatrixX3::kCols == 1, "X3 must be 3x1");
    static_assert(TMatrixBary::kRows == 3 and TMatrixBary::kCols == 1, "bary must be 3x1");
    static_assert(TMatrixC::kRows == 3 and TMatrixC::kCols == 1, "C must be 3x1");
    using ScalarType = typename TMatrixP::ScalarType;
    using Vec3       = math::linalg::mini::SVector<ScalarType, 3>;
    // C(x) = [N T B]^T ([x_i x_j x_k] bary - P)
    Vec3 X = bary(0) * X1 + bary(1) * X2 + bary(2) * X3;
    C      = NTB.Transpose() * (X - P);
}

/**
 * @brief Computes constraint \f$ C(x) = \begin{bmatrix} N & T & B \end{bmatrix}^T
 * (\begin{bmatrix} x_i & x_j & x_k \end{bmatrix} \beta - P) \in \mathbb{R}^3 \f$ and its gradient.
 *
 * @tparam TMatrixP Matrix type for \f$ P \f$
 * @tparam TMatrixNTB Matrix type for \f$ \begin{bmatrix} N & T & B \end{bmatrix} \f$
 * @tparam TMatrixX1 Matrix type for \f$ x_i \f$
 * @tparam TMatrixX2 Matrix type for \f$ x_j \f$
 * @tparam TMatrixX3 Matrix type for \f$ x_k \f$
 * @tparam TMatrixBary Matrix type for \f$ bary \f$
 * @tparam TMatrixC Matrix type for \f$ C \f$
 * @tparam TMatrixGradCT Matrix type for \f$ \nabla C \f$
 * @param P `3 x 1` origin of the contact basis
 * @param NTB `3 x 3` contact basis (normal, tangent, bitangent)
 * @param X1 `3 x 1` vertex position \f$ x_i \f$
 * @param X2 `3 x 1` vertex position \f$ x_j \f$
 * @param X3 `3 x 1` vertex position \f$ x_k \f$
 * @param bary `3 x 1` barycentric coordinates of the contact point on the triangle
 * @param C `3 x 1` constraint (vector) value
 * @param gradCT `9 x 3` gradient of the constraint w.r.t. triangle vertex positions
 */
template <
    math::linalg::mini::CMatrix TMatrixP,
    math::linalg::mini::CMatrix TMatrixNTB,
    math::linalg::mini::CMatrix TMatrixX1,
    math::linalg::mini::CMatrix TMatrixX2,
    math::linalg::mini::CMatrix TMatrixX3,
    math::linalg::mini::CMatrix TMatrixBary,
    math::linalg::mini::CMatrix TMatrixC,
    math::linalg::mini::CMatrix TMatrixGradCT>
PBAT_HOST_DEVICE void TriangleEnvironmentConstraintAndGradient(
    TMatrixP const& P,
    TMatrixNTB const& NTB,
    TMatrixX1 const& X1,
    TMatrixX2 const& X2,
    TMatrixX3 const& X3,
    TMatrixBary const& bary,
    TMatrixC& C,
    TMatrixGradCT& gradCT)
{
    static_assert(TMatrixGradCT::kRows == 9 and TMatrixGradCT::kCols == 3, "gradCT must be 9x3");
    using math::linalg::mini::SMatrix;
    TriangleEnvironmentConstraint(P, NTB, X1, X2, X3, bary, C);
    // \nabla_{x_i,x_j,x_k} C_c = \beta \otimes NTB[:,c]
    common::ForRange<0, 3>(
        [&]<auto c>() {
            common::ForRange<0, 3>(
                [&]<auto i>() {
                    gradCT.Col(c).template Slice<3, 1>(i * 3, 0) = bary(i) * NTB.Col(c);
                });
        });
}
} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_CONSTRAINTS_H