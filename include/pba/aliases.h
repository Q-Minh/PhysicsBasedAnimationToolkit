#ifndef PBA_CORE_ALIASES_H
#define PBA_CORE_ALIASES_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstddef>

namespace pba {

using Index  = std::ptrdiff_t;
using Scalar = double;

template <Index N>
using Vector = Eigen::Vector<Scalar, N>;
template <Index Rows, Index Cols>
using Matrix = Eigen::Matrix<Scalar, Rows, Cols>;

using VectorX = Eigen::Vector<Scalar, Eigen::Dynamic>;
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <Index N>
using IndexVector = Eigen::Vector<Index, N>;

template <Index Rows, Index Cols>
using IndexMatrix = Eigen::Matrix<Index, Rows, Cols>;

using IndexVectorX = Eigen::Vector<Index, Eigen::Dynamic>;
using IndexMatrixX = Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic>;

using CSCMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
using CSRMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor>;

} // namespace pba

#endif // PBA_CORE_ALIASES_H