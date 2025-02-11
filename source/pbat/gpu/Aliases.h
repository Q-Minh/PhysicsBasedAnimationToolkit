/**
 * @file Aliases.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Type aliases for GPU code
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GPU_ALIASES_H
#define PBAT_GPU_ALIASES_H

#include <Eigen/Core>
#include <cstdint>

namespace pbat {

using GpuScalar = float;        ///< Scalar type for GPU code
using GpuIndex  = std::int32_t; ///< Index type for GPU code

using GpuMatrixX =
    Eigen::Matrix<GpuScalar, Eigen::Dynamic, Eigen::Dynamic>; ///< Matrix type for GPU code
using GpuIndexMatrixX =
    Eigen::Matrix<GpuIndex, Eigen::Dynamic, Eigen::Dynamic>; ///< Index matrix type for GPU code

using GpuVectorX      = Eigen::Vector<GpuScalar, Eigen::Dynamic>; ///< Vector type for GPU code
using GpuIndexVectorX = Eigen::Vector<GpuIndex, Eigen::Dynamic>; ///< Index vector type for GPU code

} // namespace pbat

#endif // PBAT_GPU_ALIASES_H
