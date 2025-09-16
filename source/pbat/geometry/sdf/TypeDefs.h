/**
 * @file TypeDefs.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file defines common type definitions used in the SDF module
 * @version 0.1
 * @date 2025-09-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef PBAT_GEOMETRY_SDF_TYPEDEFS_H
#define PBAT_GEOMETRY_SDF_TYPEDEFS_H

#include "pbat/common/Concepts.h"
#include "pbat/math/linalg/mini/Matrix.h"

namespace pbat::geometry::sdf {

template <common::CArithmetic TScalar>
using Vec2 = math::linalg::mini::SVector<TScalar, 2>; ///< 2D vector type

template <common::CArithmetic TScalar>
using Vec3 = math::linalg::mini::SVector<TScalar, 3>; ///< 3D vector type

template <common::CArithmetic TScalar>
using Mat2 = math::linalg::mini::SMatrix<TScalar, 2, 2>; ///< 2x2 matrix type

template <common::CArithmetic TScalar>
using Mat3 = math::linalg::mini::SMatrix<TScalar, 3, 3>; ///< 3x3 matrix type

template <common::CArithmetic TScalar>
using Zero2 = math::linalg::mini::Zeros<TScalar, 2, 1>; ///< 2D zero vector type

template <common::CArithmetic TScalar>
using Zero3 = math::linalg::mini::Zeros<TScalar, 3, 1>; ///< 3D zero vector type

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_TYPEDEFS_H
