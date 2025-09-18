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
 * @brief Base struct for all primitive shapes
 */
struct Primitive
{
};

/**
 * @brief Sphere centered in \f$ (0,0,0) \f$ with radius \f$ R \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Sphere : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType R;               ///< Sphere radius
    /**
     * @brief Default constructor
     */
    Sphere() = default;
    /**
     * @brief Construct a new Sphere object
     * @param R_ Sphere radius
     */
    explicit Sphere(ScalarType R_) : R(R_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the sphere (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const { return Norm(p) - R; }
};

/**
 * @brief Axis-aligned box centered in \f$ (0,0,0) \f$ with half extents \f$ \text{he} \in
 * \mathbb{R}^3 \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Box : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> he;        ///< Half extents of the box along each axis
    /**
     * @brief Default constructor
     */
    Box() = default;
    /**
     * @brief Construct a new Box object
     * @param he_ Half extents of the box along each axis
     */
    explicit Box(Vec3<ScalarType> const& he_) : he(he_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the box (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
    {
        Vec3<ScalarType> q = Abs(p) - he;
        Zero3<ScalarType> constexpr zero3{};
        using namespace std;
        return Norm(Max(q, zero3)) + min(max(q(0), max(q(1), q(2))), ScalarType(0));
    }
};

/**
 * @brief Box frame with half extents \f$ \text{he} \in \mathbb{R}^3 \f$ and thickness \f$ t \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct BoxFrame : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> he;        ///< Half extents of the box frame along each axis
    ScalarType t;               ///< Thickness of the box frame
    /**
     * @brief Default constructor
     */
    BoxFrame() = default;
    /**
     * @brief Construct a new Box Frame object
     * @param he_ Half extents of the box frame along each axis
     * @param t_ Thickness of the box frame
     */
    explicit BoxFrame(Vec3<ScalarType> const& he_, ScalarType t_) : he(he_), t(t_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the box frame (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> p) const
    {
        p                  = Abs(p) - he;
        Vec3<ScalarType> q = Abs(p + t) - t;
        Zero3<ScalarType> constexpr zero3{};
        ScalarType constexpr zero{0};
        using namespace std;
        // clang-format off
        return min(
            min(
                Norm(Max(Vec3<ScalarType>{p(0),q(1),q(2)},zero3)) + min(max(p(0),max(q(1),q(2))), zero),
                Norm(Max(Vec3<ScalarType>{q(0),p(1),q(2)},zero3)) + min(max(q(0),max(p(1),q(2))), zero)
            ),
            Norm(Max(Vec3<ScalarType>{q(0),q(1),p(2)},zero3)) + min(max(q(0),max(q(1),p(2))), zero)
        );
        // clang-format on
    }
};

/**
 * @brief Torus centered in \f$ (0,0,0) \f$ with minor+major radius \f$ t = (r,R) \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Torus : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> t;         ///< t[0]: minor radius, t[1]: major radius
    /**
     * @brief Default constructor
     */
    Torus() = default;
    /**
     * @brief Construct a new Torus object
     * @param t_ Minor and major radius of the torus
     */
    explicit Torus(Vec2<ScalarType> const& t_) : t(t_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the torus (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
struct CappedTorus : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> sc;        ///< Probably minor+major radius
    ScalarType ra;              ///< Unknown
    ScalarType rb;              ///< Unknown
    /**
     * @brief Default constructor
     */
    CappedTorus() = default;
    /**
     * @brief Construct a new Capped Torus object
     * @param sc_ Probably minor+major radius
     * @param ra_ Unknown
     * @param rb_ Unknown
     */
    explicit CappedTorus(Vec2<ScalarType> const& sc_, ScalarType ra_, ScalarType rb_)
        : sc(sc_), ra(ra_), rb(rb_)
    {
    }
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capped torus (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> p) const
    {
        using namespace std;
        p(0)          = abs(p(0));
        bool const bk = sc(1) * p(0) > sc(0) * p(1);
        auto pxy      = p.Slice<2, 1>(0, 0);
        // NOTE: Not sure if better to do branchless but compute a norm (i.e. expensive sqrt), or
        // use ternary operator and add divergent branching, but save the sqrt when possible.
        ScalarType k = bk * (Dot(pxy, sc)) + (not bk) * Norm(pxy);
        return sqrt(SquaredNorm(p) + ra * ra - ScalarType(2) * ra * k) - rb;
    }
};

/**
 * @brief Link shape as elongated torus with elongation length \f$ le \f$ and minor+major radius \f$
 * t = (r,R) \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Link : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> t;         ///< t[0]: minor radius, t[1]: major radius
    ScalarType le;              ///< Elongation length
    /**
     * @brief Default constructor
     */
    Link() = default;
    /**
     * @brief Construct a new Link object
     * @param t_ Minor and major radius of the link
     * @param le_ Elongation length of the link
     */
    explicit Link(Vec2<ScalarType> const& t_, ScalarType le_) : t(t_), le(le_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the link shape (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        ScalarType constexpr zero{0};
        Vec3<ScalarType> q = Vec3<ScalarType>{p(0), max(abs(p(1)) - le, zero), p(2)};
        auto qxy           = q.Slice<2, 1>(0, 0);
        return Norm(Vec2<ScalarType>{Norm(qxy) - t(0), q(2)}) - t(1);
    }
};

/**
 * @brief Infinite cylinder
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct InfiniteCylinder : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> c; ///< Center of the cylinder (on the axis) in c(0), c(1) and radius in c(2)
    /**
     * @brief Default constructor
     */
    InfiniteCylinder() = default;
    /**
     * @brief Construct a new Infinite Cylinder object
     * @param c_ Center of the cylinder (on the axis) in c(0), c(1) and radius in c(2)
     */
    explicit InfiniteCylinder(Vec3<ScalarType> const& c_) : c(c_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the infinite cylinder (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
struct Cone : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> c;         ///< sin/cos of the angle
    ScalarType h;               ///< Height of the cone
    /**
     * @brief Default constructor
     */
    Cone() = default;
    /**
     * @brief Construct a new Cone object
     * @param c_ sin/cos of the angle
     * @param h_ Height of the cone
     */
    explicit Cone(Vec2<ScalarType> const& c_, ScalarType h_) : c(c_), h(h_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the cone (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        ScalarType constexpr zero{0};
        ScalarType constexpr one{1};
        Vec2<ScalarType> q = h * Vec2<ScalarType>{c(0) / c(1), ScalarType(-1)};
        Vec2<ScalarType> w{Norm(Vec2<ScalarType>{p(0), p(2)}), p(1)};
        Vec2<ScalarType> a = w - q * clamp(Dot(w, q) / Dot(q, q), zero, one);
        Vec2<ScalarType> b = w - q * Vec2<ScalarType>{clamp(w(0) / q(0), zero, one), one};
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
struct InfiniteCone : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> c;         ///< sin/cos of the angle
    /**
     * @brief Default constructor
     */
    InfiniteCone() = default;
    /**
     * @brief Construct a new Infinite Cone object
     * @param c_ sin/cos of the angle
     */
    explicit InfiniteCone(Vec2<ScalarType> const& c_) : c(c_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the infinite cone (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
struct Plane : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the plane (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
struct HexagonalPrism : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec2<ScalarType> h;         ///< h[0]: radius of the hexagon, h[1]: half height of the prism
    /**
     * @brief Default constructor
     */
    HexagonalPrism() = default;
    /**
     * @brief Construct a new Hexagonal Prism object
     * @param h_ h[0]: radius of the hexagon, h[1]: half height of the prism
     */
    explicit HexagonalPrism(Vec2<ScalarType> const& h_) : h(h_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the hexagonal prism (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> p) const
    {
        using namespace std;
        Vec3<ScalarType> const k{ScalarType{-0.8660254}, ScalarType{0.5}, ScalarType{0.57735}};
        p        = Abs(p);
        auto pxy = p.Slice<2, 1>(0, 0);
        auto kxy = k.Slice<2, 1>(0, 0);
        ScalarType constexpr zero{0};
        pxy -= ScalarType(2) * min(Dot(kxy, pxy), zero) * kxy;
        Vec2<ScalarType> d = Vec2<ScalarType>{
            Norm(pxy - Vec2<ScalarType>{clamp(p(0), -k(2) * h(0), k(2) * h(0)), h(0)}) *
                sign(p(1) - h(0)),
            p(2) - h(1)};
        return min(max(d(0), d(1)), zero) + Norm(Max(d, Zero2<ScalarType>{}));
    }
};

/**
 * @brief Capsule shape with endpoints \f$ a, b \in \mathbb{R}^3 \f$ and radius \f$ r \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Capsule : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> a;         ///< Endpoint a of the capsule
    Vec3<ScalarType> b;         ///< Endpoint b of the capsule
    ScalarType r;               ///< Radius of the capsule
    /**
     * @brief Default constructor
     */
    Capsule() = default;
    /**
     * @brief Construct a new Capsule object
     * @param a_ Endpoint a of the capsule
     * @param b_ Endpoint b of the capsule
     * @param r_ Radius of the capsule
     */
    explicit Capsule(Vec3<ScalarType> const& a_, Vec3<ScalarType> const& b_, ScalarType r_)
        : a(a_), b(b_), r(r_)
    {
    }
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capsule (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
struct VerticalCapsule : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the capsule
    ScalarType r;               ///< Radius of the capsule
    /**
     * @brief Default constructor
     */
    VerticalCapsule() = default;
    /**
     * @brief Construct a new Vertical Capsule object
     * @param h_ Height of the capsule
     * @param r_ Radius of the capsule
     */
    explicit VerticalCapsule(ScalarType h_, ScalarType r_) : h(h_), r(r_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capsule (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> p) const
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
struct CappedCylinder : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> a;         ///< Endpoint a of the capped cylinder
    Vec3<ScalarType> b;         ///< Endpoint b of the capped cylinder
    ScalarType r;               ///< Radius of the capped cylinder
    /**
     * @brief Default constructor
     */
    CappedCylinder() = default;
    /**
     * @brief Construct a new Capped Cylinder object
     * @param a_ Endpoint a of the capped cylinder
     * @param b_ Endpoint b of the capped cylinder
     * @param r_ Radius of the capped cylinder
     */
    explicit CappedCylinder(Vec3<ScalarType> const& a_, Vec3<ScalarType> const& b_, ScalarType r_)
        : a(a_), b(b_), r(r_)
    {
    }
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capped cylinder (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
struct VerticalCappedCylinder : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the capped cylinder
    ScalarType r;               ///< Radius of the capped cylinder
    /**
     * @brief Default constructor
     */
    VerticalCappedCylinder() = default;
    /**
     * @brief Construct a new Vertical Capped Cylinder object
     * @param h_ Height of the capped cylinder
     * @param r_ Radius of the capped cylinder
     */
    explicit VerticalCappedCylinder(ScalarType h_, ScalarType r_) : h(h_), r(r_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the vertical capped cylinder (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        Vec2<ScalarType> pxz{p(0), p(2)};
        Vec2<ScalarType> d = Abs(Vec2<ScalarType>{Norm(pxz), p(1)}) - Vec2<ScalarType>{r, h};
        ScalarType constexpr zero{0};
        return min(max(d(0), d(1)), zero) + Norm(Max(d, Zero2<ScalarType>{}));
    }
};

/**
 * @brief Rounded cylinder shape with height \f$ h \f$, radius \f$ \text{ra} \f$ and rounding radius
 * \f$ \text{rb} \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct RoundedCylinder : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the rounded cylinder
    ScalarType ra;              ///< Radius of the rounded cylinder
    ScalarType rb;              ///< Rounding radius at edges
    /**
     * @brief Default constructor
     */
    RoundedCylinder() = default;
    /**
     * @brief Construct a new Rounded Cylinder object
     * @param h_ Height of the rounded cylinder
     * @param ra_ Radius of the rounded cylinder
     * @param rb_ Rounding radius at edges
     */
    explicit RoundedCylinder(ScalarType h_, ScalarType ra_, ScalarType rb_)
        : h(h_), ra(ra_), rb(rb_)
    {
    }
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the rounded cylinder (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
    {
        using namespace std;
        Vec2<ScalarType> pxz{p(0), p(2)};
        Vec2<ScalarType> d{Norm(pxz) - ScalarType(2) * ra + rb, abs(p(1)) - h};
        ScalarType constexpr zero{0};
        Zero2<ScalarType> constexpr zero2{};
        return min(max(d(0), d(1)), zero) + Norm(Max(d, zero2)) - rb;
    }
};

/**
 * @brief Capped cone shape with height \f$ h \f$ and minor+major radius \f$ r^1, r^2 \f$
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct VerticalCappedCone : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the capped cone
    ScalarType r1;              ///< Minor radius of the capped cone
    ScalarType r2;              ///< Major radius of the capped cone
    /**
     * @brief Default constructor
     */
    VerticalCappedCone() = default;
    /**
     * @brief Construct a new Vertical Capped Cone object
     * @param h_ Height of the capped cone
     * @param r1_ Minor radius of the capped cone
     * @param r2_ Major radius of the capped cone
     */
    explicit VerticalCappedCone(ScalarType h_, ScalarType r1_, ScalarType r2_)
        : h(h_), r1(r1_), r2(r2_)
    {
    }
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the capped cone (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
struct CutHollowSphere : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType r;               ///< Radius of the hollow sphere
    ScalarType h;               ///< Cut height
    ScalarType t;               ///< Thickness of the hollow sphere
    /**
     * @brief Default constructor
     */
    CutHollowSphere() = default;
    /**
     * @brief Construct a new Cut Hollow Sphere object
     * @param r_ Radius of the hollow sphere
     * @param h_ Cut height
     * @param t_ Thickness of the hollow sphere
     */
    explicit CutHollowSphere(ScalarType r_, ScalarType h_, ScalarType t_) : r(r_), h(h_), t(t_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the cut hollow sphere (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
struct VerticalRoundCone : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the round cone
    ScalarType r1;              ///< Radius at the bottom of the round cone
    ScalarType r2;              ///< Radius at the top of the round cone
    /**
     * @brief Default constructor
     */
    VerticalRoundCone() = default;
    /**
     * @brief Construct a new Vertical Round Cone object
     * @param h_ Height of the round cone
     * @param r1_ Radius at the bottom of the round cone
     * @param r2_ Radius at the top of the round cone
     */
    explicit VerticalRoundCone(ScalarType h_, ScalarType r1_, ScalarType r2_)
        : h(h_), r1(r1_), r2(r2_)
    {
    }
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the vertical round cone (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
            return Norm(q - Vec2<ScalarType>{zero, h}) - r2;
        return Dot(q, Vec2<ScalarType>{a, b}) - r1;
    }
};

/**
 * @brief Octahedron shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Octahedron : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType s;               ///< Size of the octahedron
    /**
     * @brief Default constructor
     */
    Octahedron() = default;
    /**
     * @brief Construct a new Octahedron object
     * @param s_ Size of the octahedron
     */
    explicit Octahedron(ScalarType s_) : s(s_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the octahedron (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> p) const
    {
        using namespace std;
        p            = Abs(p);
        ScalarType m = p(0) + p(1) + p(2) - s;
        ScalarType constexpr three{3};
        Vec3<ScalarType> q;
        if (three * p(0) < m)
            q = p;
        else if (three * p(1) < m)
            q = Vec3<ScalarType>{p(1), p(2), p(0)};
        else if (three * p(2) < m)
            q = Vec3<ScalarType>{p(2), p(0), p(1)};
        else
            return m * ScalarType(0.57735027);
        ScalarType constexpr zero{0};
        ScalarType k = clamp(ScalarType(0.5) * (q(2) - q(1) + s), zero, s);
        return Norm(Vec3<ScalarType>{q(0), q(1) - s + k, q(2) - k});
    }
};

/**
 * @brief Pyramid shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Pyramid : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    ScalarType h;               ///< Height of the pyramid
    /**
     * @brief Default constructor
     */
    Pyramid() = default;
    /**
     * @brief Construct a new Pyramid object
     * @param h_ Height of the pyramid
     */
    explicit Pyramid(ScalarType h_) : h(h_) {}
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the pyramid (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> p) const
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

        ScalarType s = max(-q(0), zero);
        ScalarType t = clamp((q(1) - half * p(2)) / (m2 + quarter), zero, one);

        ScalarType a = m2 * (q(0) + s) * (q(0) + s) + q(1) * q(1);
        ScalarType b =
            m2 * (q(0) + half * t) * (q(0) + half * t) + (q(1) - m2 * t) * (q(1) - m2 * t);

        bool bd2      = min(q(1), -q(0) * m2 - q(1) * half) > zero;
        ScalarType d2 = bd2 * zero + (not bd2) * min(a, b);

        return sqrt((d2 + q(2) * q(2)) / m2) * sign(max(q(2), -p(1)));
    }
};

/**
 * @brief Triangle shape
 * @tparam TScalar Scalar type
 */
template <common::CArithmetic TScalar>
struct Triangle : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> a;         ///< Vertex a of the triangle
    Vec3<ScalarType> b;         ///< Vertex b of the triangle
    Vec3<ScalarType> c;         ///< Vertex c of the triangle
    /**
     * @brief Default constructor
     */
    Triangle() = default;
    /**
     * @brief Construct a new Triangle object
     * @param a_ Vertex a of the triangle
     * @param b_ Vertex b of the triangle
     * @param c_ Vertex c of the triangle
     */
    explicit Triangle(
        Vec3<ScalarType> const& a_,
        Vec3<ScalarType> const& b_,
        Vec3<ScalarType> const& c_)
        : a(a_), b(b_), c(c_)
    {
    }
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the triangle (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
        bool bs =
            (sign(Dot(Cross(ba, nor), pa)) + sign(Dot(Cross(cb, nor), pb)) +
                 sign(Dot(Cross(ac, nor), pc)) <
             two);
        return sqrt(
            bs ? min(min(SquaredNorm(ba * clamp(Dot(ba, pa) / SquaredNorm(ba), zero, one) - pa),
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
struct Quadrilateral : public Primitive
{
    using ScalarType = TScalar; ///< Scalar type
    Vec3<ScalarType> a;         ///< Vertex a of the quadrilateral
    Vec3<ScalarType> b;         ///< Vertex b of the quadrilateral
    Vec3<ScalarType> c;         ///< Vertex c of the quadrilateral
    Vec3<ScalarType> d;         ///< Vertex d of the quadrilateral
    /**
     * @brief Default constructor
     */
    Quadrilateral() = default;
    /**
     * @brief Construct a new Quadrilateral object
     * @param a_ Vertex a of the quadrilateral
     * @param b_ Vertex b of the quadrilateral
     * @param c_ Vertex c of the quadrilateral
     * @param d_ Vertex d of the quadrilateral
     */
    explicit Quadrilateral(
        Vec3<ScalarType> const& a_,
        Vec3<ScalarType> const& b_,
        Vec3<ScalarType> const& c_,
        Vec3<ScalarType> const& d_)
        : a(a_), b(b_), c(c_), d(d_)
    {
    }
    /**
     * @brief Evaluate the signed distance function at a point
     * @param p Point in 3D space
     * @return Signed distance to the quadrilateral (negative inside, positive outside)
     */
    PBAT_HOST_DEVICE ScalarType Eval(Vec3<ScalarType> const& p) const
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
        bool bs =
            (sign(Dot(Cross(ba, nor), pa)) + sign(Dot(Cross(cb, nor), pb)) +
                 sign(Dot(Cross(dc, nor), pc)) + sign(Dot(Cross(ad, nor), pd)) <
             three);
        return sqrt(
            bs ?
                min(min(min(SquaredNorm(ba * clamp(Dot(ba, pa) / SquaredNorm(ba), zero, one) - pa),
                            SquaredNorm(cb * clamp(Dot(cb, pb) / SquaredNorm(cb), zero, one) - pb)),
                        SquaredNorm(dc * clamp(Dot(dc, pc) / SquaredNorm(dc), zero, one) - pc)),
                    SquaredNorm(ad * clamp(Dot(ad, pd) / SquaredNorm(ad), zero, one) - pd)) :
                Dot(nor, pa) * Dot(nor, pa) / SquaredNorm(nor));
    }
};

} // namespace pbat::geometry::sdf

#endif // PBAT_GEOMETRY_SDF_PRIMITIVE_H
