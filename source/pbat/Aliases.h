#ifndef PBAT_ALIASES_H
#define PBAT_ALIASES_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstddef>

/**
 * @file
 */

/**
 * @brief
 */
namespace pbat {

using Index = std::ptrdiff_t; ///< Index type
using Scalar = double; ///< Scalar type
/**
 * @brief Fixed-size vector type
 * @tparam N
 */
template <Index N>
using Vector = Eigen::Vector<Scalar, N>;
/**
 * @brief Fixed-size matrix type
 * @tparam Rows
 * @tparam Cols
 */
template <Index Rows, Index Cols>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols>;

using VectorX = Eigen::Vector<Scalar, Eigen::Dynamic>; ///< Dynamic-size vector type
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>; ///< Dynamic-size matrix type
/**
 * @brief Fixed-size index vector type
 * @tparam N
 */
template <Index N>
using IndexVector = Eigen::Vector<Index, N>;
/**
 * @brief Fixed-size index matrix type
 * @tparam Rows
 * @tparam Cols
 */
template <Index Rows, Index Cols>
using IndexMatrix = Eigen::Matrix<Index, Rows, Cols>;

using IndexVectorX = Eigen::Vector<Index, Eigen::Dynamic>; ///< Dynamic-size index vector type
using IndexMatrixX = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic>; ///< Dynamic-size index matrix type

using CSCMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>; ///< Column-major sparse matrix type
using CSRMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>; ///< Row-major sparse matrix type

} // namespace pbat

#endif // PBAT_ALIASES_H