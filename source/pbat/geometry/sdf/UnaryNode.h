/**
 * @file UnaryNode.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file defines unary nodes for SDF compositions
 * @version 0.1
 * @date 2025-09-16
 * @details Credits go to https://iquilezles.org/articles/distfunctions/, thank you Inigo Quilez!
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_GEOMETRY_SDF_UNARYNODE_H
#define PBAT_GEOMETRY_SDF_UNARYNODE_H

#include "TypeDefs.h"
#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"
#include "pbat/math/linalg/mini/BinaryOperations.h"
#include "pbat/math/linalg/mini/UnaryOperations.h"

#include <algorithm>
#include <cmath>

namespace pbat::geometry::sdf {

/**
 * @brief Base struct for all unary nodes
 */
struct UnaryNode
{
};

/**
 * @brief Elongation operation along the axes
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Elongate : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> h;         ///< Elongation vector
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the elongated shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        Vec3<ScalarType> q = Abs(p) - h;
        using namespace std;
        return sdf.eval(Max(q, Zero3<ScalarType>{})) +
               min(max(q(0), max(q(1), q(2))), ScalarType(0));
    }
};

/**
 * @brief Rounding operation (i.e. positive offset surface)
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Round : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType r;               ///< Rounding radius
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the rounded shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        return sdf.eval(p) - r;
    }
};

/**
 * @brief Onion operation (i.e. carving interior)
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Onion : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType t;               ///< Onion thickness
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the onioned shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        using namespace std;
        return abs(sdf.eval(p)) - t;
    }
};

/**
 * @brief Symmetrization operation along the x axis
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Symmetrize : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the symmetrized shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p, FSdf&& sdf) const
    {
        using namespace std;
        p(0) = abs(p(0));
        p(2) = abs(p(2));
        return sdf.eval(p);
    }
};

/**
 * @brief Grid-like repetition operation along the axes
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Repeat : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType s;               ///< Uniform repetition spacing
    Vec3<ScalarType> l;         ///< Half number of repetitions along each axis
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the repeated shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p, FSdf&& sdf) const
    {
        using namespace std;
        Vec3<ScalarType> pc{
            clamp(round(p(0) / s), -l(0), l(0)),
            clamp(round(p(1) / s), -l(1), l(1)),
            clamp(round(p(2) / s), -l(2), l(2))};
        Vec3<ScalarType> q = p - s * pc;
        return sdf.eval(q);
    }
};

template <common::CArithmetic TScalar>
struct PeriodicWaveDisplace : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> f;         ///< Frequency along each axis
    Vec3<ScalarType> g;         ///< Amplitude of the wave displacement
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the wave-displaced shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p, FSdf&& sdf) const
    {
        using namespace std;
        // clang-format off
        ScalarType d = 
            g(0)*sin(f(0)*p(0)) * 
            g(1)*sin(f(1)*p(1)) * 
            g(2)*sin(f(2)*p(2));
        // clang-format on
        return sdf.eval(p) + d;
    }
};

/**
 * @brief Twist operation around the y axis
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Twist : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType k;               ///< Twist factor
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the twisted shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p, FSdf&& sdf) const
    {
        using namespace std;
        ScalarType c = cos(k * p(1));
        ScalarType s = sin(k * p(1));
        Vec3 q{c * p(0) - s * p(2), s * p(0) + c * p(2), p(1)};
        return sdf.eval(q);
    }
};

/**
 * @brief Bend operation around the z axis
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Bend : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType k;               ///< Bend factor
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the bent shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p, FSdf&& sdf) const
    {
        using namespace std;
        ScalarType c = cos(k * p.x);
        ScalarType s = sin(k * p.x);
        Vec3 q{c * p(0) - s * p(1), s * p(0) + c * p(1), p(2)};
        return sdf.eval(q);
    }
};

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_UNARYNODE_H
