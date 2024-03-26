#ifndef PBA_CORE_ALIASES_H
#define PBA_CORE_ALIASES_H

#include <Eigen/Core>
#include <cstddef>

namespace pba {

using Index  = std::ptrdiff_t;
using Scalar = double;

using VectorX = Eigen::Vector<Scalar, Eigen::Dynamic>;
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <Index N>
using IndexVector = Eigen::Vector<Index, N>;

} // namespace pba

#endif // PBA_CORE_ALIASES_H