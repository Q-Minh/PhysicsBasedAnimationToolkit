/**
 * @file Transform.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file defines transforms useful for moving SDFs around
 * @version 0.1
 * @date 2025-09-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef PBAT_GEOMETRY_SDF_TRANSFORM_H
#define PBAT_GEOMETRY_SDF_TRANSFORM_H

#include "TypeDefs.h"
#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"
#include "pbat/math/linalg/mini/Eigen.h"

#include <Eigen/SVD>

namespace pbat::geometry::sdf {

template <common::CArithmetic TScalar>
struct Transform
{
    using ScalarType = TScalar; ///< Scalar type
    Mat3<ScalarType> R;         ///< Rotation matrix
    Vec3<ScalarType> t;         ///< Translation vector

    PBAT_HOST_DEVICE Vec3<ScalarType> operator()(Vec3<ScalarType> const& p) const
    {
        return R * p + t;
    }
    PBAT_HOST_DEVICE Vec3<ScalarType> operator/(Vec3<ScalarType> const& p) const
    {
        return R.Transpose() * (p - t);
    }
    void CleanRotation()
    {
        Eigen::Matrix<ScalarType, 3, 3> Reig = math::linalg::mini::ToEigen(R);
        Eigen::JacobiSVD<decltype(Reig)> svd{Reig, Eigen::ComputeFullU | Eigen::ComputeFullV};
        Reig = svd.matrixU() * svd.matrixV().transpose();
        R    = math::linalg::mini::FromEigen(Reig);
    }
};

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_TRANSFORM_H
