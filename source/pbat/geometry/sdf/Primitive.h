/**
 * @file Primitive.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file defines primitive SDF shapes
 * @version 0.1
 * @date 2025-09-16
 *
 * @copyright Copyright (c) 2025
 *
 * @details Credits go to https://iquilezles.org/articles/distfunctions/, thank you Inigo Quilez!
 */
#ifndef PBAT_GEOMETRY_SDF_PRIMITIVE_H
#define PBAT_GEOMETRY_SDF_PRIMITIVE_H

#include "TypeDefs.h"
#include "pbat/HostDevice.h"
#include "pbat/common/Concepts.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <algorithm>
#include <cmath>

namespace pbat::geometry::sdf {

template <common::CArithmetic TScalar>
TScalar sign(TScalar x)
{
    return static_cast<TScalar>(x > 0) - static_cast<TScalar>(x < 0);
};

/**
 * @brief Sphere centered in \f$ (0,0,0) \f$ with radius \f$ R \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Sphere
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType R;               ///< Sphere radius
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the sphere (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const { return Norm(p) - R; }
};

/**
 * @brief Axis-aligned box centered in \f$ (0,0,0) \f$ with half extents \f$ \text{he} \in
 * \mathbb{R}^3 \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Box
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> he;        ///< Half extents of the box along each axis
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the box (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        Vec3<ScalarType> q = Abs(p) - he;
        Zero3<ScalarType> constexpr zero3{};
        return Norm(Max(q, zero3)) + Min(Max(q(0), Max(q(1), q(2))), ScalarType(0));
    }
};

/**
 * @brief Box frame with half extents \f$ \text{he} \in \mathbb{R}^3 \f$ and thickness \f$ t \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct BoxFrame
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> he;        ///< Half extents of the box frame along each axis
    ScalarType t;               ///< Thickness of the box frame
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the box frame (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p) const
    {
        Vec3<ScalarType> p = Abs(p) - he;
        Vec3<ScalarType> q = Abs(p + t) - t;
        Zero3<ScalarType> constexpr zero3{};
        ScalarType constexpr zero{0};
        // clang-format off
        return Min(
            Min(
                Norm(Max(Vec3<ScalarType>{p(0),q(1),q(2)},zero3)) + Min(Max(p(0),Max(q(1),q(2))), zero),
                Norm(Max(Vec3<ScalarType>{q(0),p(1),q(2)},zero3)) + Min(Max(q(0),Max(p(1),q(2))), zero)
            ),
            Norm(Max(Vec3<ScalarType>{q(0),q(1),p(2)},zero3)) + Min(Max(q(0),Max(q(1),p(2))), zero)
        );
        // clang-format on
    }
};

/**
 * @brief Torus centered in \f$ (0,0,0) \f$ with minor+major radius \f$ t = (r,R) \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Torus
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> t;         ///< t[0]: minor radius, t[1]: major radius
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the torus (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        Vec2<ScalarType> q = Vec2<ScalarType>{Norm(Vec2<ScalarType>{p(0), p(2)}) - t(0), p(1)};
        return Norm(q) - t(1);
    }
};

/**
 * @brief Capped torus centered in \f$ (0,0,0) \f$ with parameters `sc, ra, rb`.
 * @note I don't know what the parameters mean.
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct CappedTorus
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> sc;        ///< Probably minor+major radius
    ScalarType ra;              ///< Unknown
    ScalarType rb;              ///< Unknown
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capped torus (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p) const
    {
        using namespace std;
        p(0)          = abs(p(0));
        bool const bk = sc(1) * p(0) > sc(0) * p(1);
        auto pxy      = p.Slice<2, 1>(0, 0);
        // NOTE: Not sure if better to do branchless but compute a norm (i.e. expensive sqrt), or
        // use ternary operator and add divergent branching, but save the sqrt when possible.
        ScalarType k = bk * (Dot(pxy, sc)) + (not bk) * Norm(pxy);
        return sqrt(SquaredNorm(p, p) + ra * ra - ScalarType(2) * ra * k) - rb;
    }
};

/**
 * @brief Link shape as elongated torus with elongation length \f$ le \f$ and minor+major radius \f$
 * t = (r,R) \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Link
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> t;         ///< t[0]: minor radius, t[1]: major radius
    ScalarType le;              ///< Elongation length
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the link shape (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        ScalarType constexpr zero{0};
        Vec3<ScalarType> q = Vec3<ScalarType>{p(0), Max(abs(p(1) - le, zero)), p(2)};
        auto qxy           = q.Slice<2, 1>(0, 0);
        return Norm(Vec2<ScalarType>{Norm(qxy) - t(0), q(2)}) - t(1);
    }
};

/**
 * @brief Infinite cylinder
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct InfiniteCylinder
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> c; ///< Center of the cylinder (on the axis) in c(0), c(1) and radius in c(2)
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the infinite cylinder (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        Vec2<ScalarType> d{p(0) - c(0), p(2) - c(1)};
        return Norm(d) - c(2);
    }
};

/**
 * @brief Cone shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Cone
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> c;         ///< sin/cos of the angle
    ScalarType h;               ///< Height of the cone
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the cone (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        ScalarType constexpr zero{0};
        ScalarType constexpr one{1};
        Vec2<ScalarType> q = h * Vec2(c(0) / c(1), ScalarType(-1));
        Vec2<ScalarType> w = Vec2(Norm(Vec2<ScalarType>{p(0), p(2)}), p(1));
        Vec2<ScalarType> a = w - q * clamp(Dot(w, q) / Dot(q, q), zero, one);
        Vec2<ScalarType> b = w - q * Vec2(clamp(w(0) / q(0), zero, one), one);
        ScalarType k       = sign(q(1));
        ScalarType d       = min(Dot(a, a), Dot(b, b));
        ScalarType s       = max(k * (w(0) * q(1) - w(1) * q(0)), k * (w(1) - q(1)));
        return sqrt(d) * sign(s);
    }
};

/**
 * @brief Infinite cone shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct InfiniteCone
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> c;         ///< sin/cos of the angle
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the infinite cone (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        ScalarType constexpr zero{0};
        ScalarType constexpr one{1};
        Vec2<ScalarType> q = Vec2<ScalarType>{Norm(Vec2<ScalarType>{p(0), p(2)}), -p(1)};
        ScalarType d       = Norm(q - max(Dot(q, c), zero) * c);
        bool bd            = (q(0) * c(1) - q(1) * c(0) < zero);
        return d * (bd * (-one) + (not bd) * one);
    }
};

/**
 * @brief Plane shape with normal \f$ n=(0,0,1) \f$ and point on the plane \f$ o=(0,0,0) \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Plane
{
    using ScalarType = TScalar; ///< Scalar type
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the plane (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        // n^T (p - o)
        return p(2);
    }
};

/**
 * @brief Hexagonal prism shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct HexagonalPrism
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> h;         ///< h[0]: radius of the hexagon, h[1]: half height of the prism
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the hexagonal prism (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p) const
    {
        using namespace std;
        Vec3<ScalarType> constexpr k{ScalarType{-0.8660254}, ScalarType{0.5}, ScalarType{0.57735}};
        p        = Abs(p);
        auto pxy = p.Slice<2, 1>(0, 0);
        auto kxy = k.Slice<2, 1>(0, 0);
        ScalarType constexpr zero{0};
        pxy -= ScalarType(2) * min(Dot(kxy, pxy), zero) * kxy;
        Vec2<ScalarType> d = Vec2(
            Norm(pxy - Vec2(clamp(p(0), -k(2) * h(0), k(2) * h(0)), h(0))) * sign(p(1) - h(0)),
            p(2) - h(1));
        return min(max(d(0), d(1)), zero) + Norm(max(d, zero));
    }
};

/**
 * @brief Capsule shape with endpoints \f$ a, b \in \mathbb{R}^3 \f$ and radius \f$ r \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Capsule
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> a;         ///< Endpoint a of the capsule
    Vec3<ScalarType> b;         ///< Endpoint b of the capsule
    ScalarType r;               ///< Radius of the capsule
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capsule (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        Vec3<ScalarType> pa = p - a;
        Vec3<ScalarType> ba = b - a;
        ScalarType h        = clamp(Dot(pa, ba) / Dot(ba, ba), ScalarType(0), ScalarType(1));
        return Norm(pa - h * ba) - r;
    }
};

/**
 * @brief Capsule shape with height \f$ h \f$ and radius \f$ r \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct VerticalCapsule
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the capsule
    ScalarType r;               ///< Radius of the capsule
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capsule (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p) const
    {
        using namespace std;
        p(1) -= clamp(p(1), ScalarType(0), h);
        return Norm(p) - r;
    }
};

/**
 * @brief Capped cylinder shape with endpoints \f$ a, b \in \mathbb{R}^3 \f$ and radius \f$ r \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct CappedCylinder
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> a;         ///< Endpoint a of the capped cylinder
    Vec3<ScalarType> b;         ///< Endpoint b of the capped cylinder
    ScalarType r;               ///< Radius of the capped cylinder
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capped cylinder (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        Vec3<ScalarType> ba = b - a;
        Vec3<ScalarType> pa = p - a;
        ScalarType baba     = Dot(ba, ba);
        ScalarType paba     = Dot(pa, ba);
        ScalarType x        = Norm(baba * pa - paba * ba) - r * baba;
        ScalarType y        = abs(paba - baba * ScalarType(0.5)) - baba * ScalarType(0.5);
        ScalarType x2       = x * x;
        ScalarType y2       = y * y * baba;
        ScalarType constexpr zero{0};
        ScalarType d = (max(x, y) < zero) ? -min(x2, y2) :
                                            (((x > zero) ? x2 : zero) + ((y > zero) ? y2 : zero));
        return sign(d) * sqrt(abs(d)) / baba;
    }
};

/**
 * @brief Vertical capped cylinder shape with height \f$ h \f$ and radius \f$ r \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct VerticalCappedCylinder
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the capped cylinder
    ScalarType r;               ///< Radius of the capped cylinder
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the vertical capped cylinder (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        Vec2<ScalarType> pxz{p(0), p(2)};
        Vec2<ScalarType> d = Abs(Vec2<ScalarType>{Norm(pxz), p(1)}) - Vec2<ScalarType>{r, h};
        ScalarType constexpr zero{0};
        return min(max(d(0), d(1)), zero) + Norm(Max(d, zero));
    }
};

/**
 * @brief Rounded cylinder shape with height \f$ h \f$, radius \f$ \text{ra} \f$ and rounding radius
 * \f$ \text{rb} \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct RoundedCylinder
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the rounded cylinder
    ScalarType ra;              ///< Radius of the rounded cylinder
    ScalarType rb;              ///< Rounding radius at edges
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the rounded cylinder (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        Vec2<ScalarType> pxz{p(0), p(2)};
        Vec2<ScalarType> d = Vec2(Norm(pxz) - ScalarType(2) * ra + rb, abs(p(1)) - h);
        ScalarType constexpr zero{0};
        Zero2<ScalarType> constexpr zero2{};
        return min(max(d.x, d.y), zero) + Norm(Max(d, zero2)) - rb;
    }
};

/**
 * @brief Capped cone shape with height \f$ h \f$ and minor+major radius \f$ r^1, r^2 \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct VerticalCappedCone
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the capped cone
    ScalarType r1;              ///< Minor radius of the capped cone
    ScalarType r2;              ///< Major radius of the capped cone
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capped cone (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        ScalarType constexpr zero{0};
        ScalarType constexpr one{1};
        Vec2<ScalarType> pxz{p(0), p(2)};
        Vec2<ScalarType> q{Norm(pxz), p(1)};
        Vec2<ScalarType> k1{r2, h};
        Vec2<ScalarType> k2{r2 - r1, ScalarType(2) * h};
        bool const brqy = (q(1) < zero);
        ScalarType rqy  = brqy * r1 + (not brqy) * r2;
        Vec2<ScalarType> ca{q(0) - min(q(0), rqy), abs(q(1)) - h};
        Vec2<ScalarType> cb = q - k1 + k2 * clamp(Dot(k1 - q, k2) / SquaredNorm(k2), zero, one);
        bool const bs       = (cb(0) < zero and ca(1) < zero);
        ScalarType s        = bs * (-one) + (not bs) * one;
        return s * sqrt(min(SquaredNorm(ca), SquaredNorm(cb)));
    }
};

/**
 * @brief Cut hollow sphere shape with radius \f$ r \f$, cut height \f$ h \f$ and thickness \f$ t
 * \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct CutHollowSphere
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType r;               ///< Radius of the hollow sphere
    ScalarType h;               ///< Cut height
    ScalarType t;               ///< Thickness of the hollow sphere
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the cut hollow sphere (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        ScalarType w = sqrt(r * r - h * h);
        Vec2<ScalarType> pxz{p(0), p(2)};
        Vec2<ScalarType> q{Norm(pxz), p(1)};
        bool b = (h * q(0) < w * q(1));
        return b * (Norm(q - Vec2<ScalarType>{w, h})) + (not b) * (abs(Norm(q) - r) - t);
    }
};

/**
 * @brief Vertical round cone shape with height \f$ h \f$, radii \f$ \text{r}^1, \text{r}^2 \f$ at
 * endpoints
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct VerticalRoundCone
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the round cone
    ScalarType r1;              ///< Radius at the bottom of the round cone
    ScalarType r2;              ///< Radius at the top of the round cone
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the vertical round cone (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        ScalarType b = (r1 - r2) / h;
        ScalarType a = sqrt(ScalarType(1) - b * b);
        Vec2<ScalarType> pxz{p(0), p(2)};
        Vec2<ScalarType> q{Norm(pxz), p(1)};
        ScalarType k = Dot(q, Vec2<ScalarType>{-b, a});
        ScalarType constexpr zero{0};
        if (k < zero)
            return Norm(q) - r1;
        if (k > a * h)
            return Norm(q - Vec2(zero, h)) - r2;
        return Dot(q, Vec2<ScalarType>{a, b}) - r1;
    }
};

/**
 * @brief Octahedron shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Octahedron
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType s;               ///< Size of the octahedron
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the octahedron (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p) const
    {
        using namespace std;
        p            = Abs(p);
        ScalarType m = p.x + p.y + p.z - s;
        ScalarType constexpr three{3};
        Vec3<ScalarType> q;
        if (three * p.x < m)
            q = p;
        else if (three * p.y < m)
            q = Vec3<ScalarType>{p(1), p(2), p(0)};
        else if (three * p.z < m)
            q = Vec3<ScalarType>{p(2), p(0), p(1)};
        else
            return m * ScalarType(0.57735027);
        ScalarType constexpr zero{0};
        ScalarType k = clamp(ScalarType(0.5) * (q(2) - q(1) + s), zero, s);
        return Norm(Vec3<ScalarType>(q(0), q(1) - s + k, q(2) - k));
    }
};

/**
 * @brief Pyramid shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Pyramid
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the pyramid
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the pyramid (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> p) const
    {
        using namespace std;
        ScalarType constexpr quarter{0.25};
        ScalarType constexpr half{0.5};
        ScalarType constexpr zero{0};
        ScalarType constexpr one{1};

        ScalarType m2 = h * h + quarter;

        ScalarType apx = abs(p(0));
        ScalarType apz = abs(p(2));
        bool bzx       = (p(2) > p(0));
        p(0)           = bzx * apz + (not bzx) * apx - half;
        p(2)           = bzx * apx + (not bzx) * apz - half;

        Vec3<ScalarType> q{p(2), h * p(1) - half * p(0), h * p(0) + half * p(1)};

        float s = max(-q(0), zero);
        float t = clamp((q(1) - half * p(2)) / (m2 + quarter), zero, one);

        float a = m2 * (q(0) + s) * (q(0) + s) + q(1) * q(1);
        float b = m2 * (q(0) + half * t) * (q(0) + half * t) + (q(1) - m2 * t) * (q(1) - m2 * t);

        bool bd2 = min(q(1), -q(0) * m2 - q(1) * half) > zero;
        float d2 = bd2 * zero + (not bd2) * min(a, b);

        return sqrt((d2 + q(2) * q(2)) / m2) * sign(max(q(2), -p(1)));
    }
};

/**
 * @brief Triangle shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Triangle
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> a;         ///< Vertex a of the triangle
    Vec3<ScalarType> b;         ///< Vertex b of the triangle
    Vec3<ScalarType> c;         ///< Vertex c of the triangle
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the triangle (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        Vec3<ScalarType> ba  = b - a;
        Vec3<ScalarType> pa  = p - a;
        Vec3<ScalarType> cb  = c - b;
        Vec3<ScalarType> pb  = p - b;
        Vec3<ScalarType> ac  = a - c;
        Vec3<ScalarType> pc  = p - c;
        Vec3<ScalarType> nor = Cross(ba, ac);
        ScalarType constexpr zero{0};
        ScalarType constexpr one{1};
        ScalarType constexpr two{2};
        bool b =
            (sign(Dot(Cross(ba, nor), pa)) + sign(Dot(Cross(cb, nor), pb)) +
                 sign(Dot(Cross(ac, nor), pc)) <
             two);
        return sqrt(
            b ? min(min(SquaredNorm(ba * clamp(Dot(ba, pa) / SquaredNorm(ba), zero, one) - pa),
                        SquaredNorm(cb * clamp(Dot(cb, pb) / SquaredNorm(cb), zero, one) - pb)),
                    SquaredNorm(ac * clamp(Dot(ac, pc) / SquaredNorm(ac), zero, one) - pc)) :
                Dot(nor, pa) * Dot(nor, pa) / SquaredNorm(nor));
    }
};

/**
 * @brief Quadrilateral shape with vertices \f$ a, b, c, d \in \mathbb{R}^3 \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Quadrilateral
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> a;         ///< Vertex a of the quadrilateral
    Vec3<ScalarType> b;         ///< Vertex b of the quadrilateral
    Vec3<ScalarType> c;         ///< Vertex c of the quadrilateral
    Vec3<ScalarType> d;         ///< Vertex d of the quadrilateral
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the quadrilateral (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        Vec3<ScalarType> ba  = b - a;
        Vec3<ScalarType> pa  = p - a;
        Vec3<ScalarType> cb  = c - b;
        Vec3<ScalarType> pb  = p - b;
        Vec3<ScalarType> dc  = d - c;
        Vec3<ScalarType> pc  = p - c;
        Vec3<ScalarType> ad  = a - d;
        Vec3<ScalarType> pd  = p - d;
        Vec3<ScalarType> nor = Cross(ba, ad);
        ScalarType constexpr zero{0};
        ScalarType constexpr one{1};
        ScalarType constexpr three{3};
        bool b =
            (sign(Dot(Cross(ba, nor), pa)) + sign(Dot(Cross(cb, nor), pb)) +
                 sign(Dot(Cross(dc, nor), pc)) + sign(Dot(Cross(ad, nor), pd)) <
             three);
        return sqrt(
            b ? min(min(min(SquaredNorm(ba * clamp(Dot(ba, pa) / SquaredNorm(ba), zero, one) - pa),
                            SquaredNorm(cb * clamp(Dot(cb, pb) / SquaredNorm(cb), zero, one) - pb)),
                        SquaredNorm(dc * clamp(Dot(dc, pc) / SquaredNorm(dc), zero, one) - pc)),
                    SquaredNorm(ad * clamp(Dot(ad, pd) / SquaredNorm(ad), zero, one) - pd)) :
                Dot(nor, pa) * Dot(nor, pa) / SquaredNorm(nor));
    }
};

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_PRIMITIVE_H
