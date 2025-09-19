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
 * @brief Uniform scaling operation
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Scale : public UnaryNode
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType s{TScalar(1)};   ///< Scaling factor
    /**
     * @brief Default constructor
     */
    Scale() = default;
    /**
     * @brief Construct a new Scale object
     * @param s_ Scaling factor
     */
    explicit Scale(ScalarType s_) : s(s_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the scaled shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        return s * sdf(Vec3<ScalarType>(p / s));
    }
};

/**
 * @brief Elongation operation along the axes
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Elongate : public UnaryNode
{
    using ScalarType = TScalar;                             ///< Scalar type
    Vec3<ScalarType> h{TScalar(0), TScalar(0), TScalar(0)}; ///< Elongation vector
    /**
     * @brief Default constructor
     */
    Elongate() = default;
    /**
     * @brief Construct a new Elongate object
     * @param h_ Elongation vector
     */
    explicit Elongate(Vec3<ScalarType> const& h_) : h(h_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the elongated shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        Vec3<ScalarType> q  = Abs(p) - h;
        Vec3<ScalarType> pp = Max(q, Zero3<ScalarType>{});
        using namespace std;
        return sdf(pp) + min(max(q(0), max(q(1), q(2))), ScalarType(0));
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
    ScalarType r{TScalar(0)};   ///< Rounding radius
    /**
     * @brief Default constructor
     */
    Round() = default;
    /**
     * @brief Construct a new Round object
     * @param r_ Rounding radius
     */
    explicit Round(ScalarType r_) : r(r_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the rounded shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        return sdf(p) - r;
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
    ScalarType t{TScalar(0)};   ///< Onion thickness
    /**
     * @brief Default constructor
     */
    Onion() = default;
    /**
     * @brief Construct a new Onion object
     * @param t_ Onion thickness
     */
    explicit Onion(ScalarType t_) : t(t_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the onioned shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        using namespace std;
        return abs(sdf(p)) - t;
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
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the symmetrized shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> p, FSdf&& sdf) const
    {
        using namespace std;
        p(0) = abs(p(0));
        p(2) = abs(p(2));
        return sdf(p);
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
    ScalarType s{TScalar(1)};   ///< Uniform repetition spacing
    Vec3<ScalarType> l{
        TScalar(5),
        TScalar(5),
        TScalar(5)}; ///< Half number of repetitions along each axis
    /**
     * @brief Default constructor
     */
    Repeat() = default;
    /**
     * @brief Construct a new Repeat object
     * @param s_ Uniform repetition spacing
     * @param l_ Half number of repetitions along each axis
     */
    explicit Repeat(ScalarType s_, Vec3<ScalarType> const& l_) : s(s_), l(l_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the repeated shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        using namespace std;
        Vec3<ScalarType> pc{
            clamp(round(p(0) / s), -l(0), l(0)),
            clamp(round(p(1) / s), -l(1), l(1)),
            clamp(round(p(2) / s), -l(2), l(2))};
        Vec3<ScalarType> q = p - s * pc;
        return sdf(q);
    }
};

/**
 * @brief Wave-like bumpiness operation along the axes
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Bump : public UnaryNode
{
    using ScalarType = TScalar;                                ///< Scalar type
    Vec3<ScalarType> f{TScalar(20), TScalar(20), TScalar(20)}; ///< Frequency along each axis
    Vec3<ScalarType> g{TScalar(1), TScalar(1), TScalar(1)}; ///< Amplitude of the wave displacement
    /**
     * @brief Default constructor
     */
    Bump() = default;
    /**
     * @brief Construct a new Bump object
     * @param f_ Frequency along each axis
     * @param g_ Amplitude of the wave displacement
     */
    explicit Bump(Vec3<ScalarType> const& f_, Vec3<ScalarType> const& g_) : f(f_), g(g_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the wave-displaced shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        using namespace std;
        // clang-format off
        ScalarType d = 
            g(0)*sin(f(0)*p(0)) * 
            g(1)*sin(f(1)*p(1)) * 
            g(2)*sin(f(2)*p(2));
        // clang-format on
        return sdf(p) + d;
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
    ScalarType k{5};            ///< Twist factor
    /**
     * @brief Default constructor
     */
    Twist() = default;
    /**
     * @brief Construct a new Twist object
     * @param k_ Twist factor
     */
    explicit Twist(ScalarType k_) : k(k_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the twisted shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        using namespace std;
        ScalarType c = cos(k * p(1));
        ScalarType s = sin(k * p(1));
        Vec3<ScalarType> q{c * p(0) - s * p(2), s * p(0) + c * p(2), p(1)};
        return sdf(q);
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
    ScalarType k{5};            ///< Bend factor
    /**
     * @brief Default constructor
     */
    Bend() = default;
    /**
     * @brief Construct a new Bend object
     * @param k_ Bend factor
     */
    explicit Bend(ScalarType k_) : k(k_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @tparam FSdf Callable type with signature `ScalarType(Vec3<ScalarType> const&)`
     * @param p `3 x 1` query point in 3D space
     * @return Signed distance to the bent shape
     */
    template <class FSdf>
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p, FSdf&& sdf) const
    {
        using namespace std;
        ScalarType c = cos(k * p(0));
        ScalarType s = sin(k * p(0));
        Vec3<ScalarType> q{c * p(0) - s * p(1), s * p(0) + c * p(1), p(2)};
        return sdf(q);
    }
};

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_UNARYNODE_H
