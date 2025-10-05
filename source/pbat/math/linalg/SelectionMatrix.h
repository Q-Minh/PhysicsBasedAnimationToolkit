#ifndef PBAT_MATH_LINALG_SELECTION_MATRIX_H
#define PBAT_MATH_LINALG_SELECTION_MATRIX_H

#include "pbat/Aliases.h"

namespace pbat {
namespace math {
namespace linalg {

/**
 * @brief Construct the selection matrix S s.t. X*S selects all columns C of X.
 *
 * @tparam TDerivedC Type of the index vector
 * @param C Vector of indices
 * @param n Number of rows in the selection matrix
 * @return `n x |C|` selection matrix
 */
template <class TDerivedC>
CSCMatrix SelectionMatrix(Eigen::DenseBase<TDerivedC> const& C, Index n = Index(-1))
{
    if (n < 0)
        n = C.maxCoeff() + 1;
    CSCMatrix S(n, C.size());
    S.reserve(IndexVectorX::Ones(C.size()));
    for (auto c = 0; c < C.size(); ++c)
        S.insert(C(c), c) = Scalar(1);
    return S;
}

} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_SELECTION_MATRIX_H