/**
 * @file MeshDistance.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Mesh distance computation types for contact mechanics
 * @version 0.1
 * @date 2026-02-18
 * @copyright Copyright (c) 2026
 */

#ifndef PBAT_GEOMETRY_MESHDISTANCE_H
#define PBAT_GEOMETRY_MESHDISTANCE_H

#include "pbat/common/Concepts.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <cmath>

namespace pbat::geometry {

/**
 * @brief Concept for mesh distance computations
 * @tparam T
 */
template <class T>
concept CMeshDistance = requires(T t) {
    { T::kStencil } -> std::convertible_to<int>;
    { T::kDims } -> std::convertible_to<int>;
    { T::kDofs } -> std::convertible_to<int>;
    requires common::CFloatingPoint<typename T::ScalarType>;
    {
        t.Eval(std::declval<math::linalg::mini::SVector<typename T::ScalarType, T::kDofs>>())
    } -> std::convertible_to<typename T::ScalarType>;
    {
        t.Gradient(std::declval<math::linalg::mini::SVector<typename T::ScalarType, T::kDofs>>())
    } -> std::convertible_to<math::linalg::mini::SVector<typename T::ScalarType, T::kDofs>>;
    {
        t.Hessian(std::declval<math::linalg::mini::SVector<typename T::ScalarType, T::kDofs>>())
    }
    -> std::convertible_to<math::linalg::mini::SMatrix<typename T::ScalarType, T::kDofs, T::kDofs>>;
};

/**
 * @brief Point-point distance computation
 *
 * Computes the Euclidean distance \f$ d = \|x - y\| \f$ and its derivatives
 * for two 3D points \f$ x \f$ and \f$ y \f$.
 *
 * @tparam TScalar Floating point scalar type
 */
template <common::CFloatingPoint TScalar>
struct PointPointDistance
{
    using ScalarType              = TScalar;
    static constexpr int kStencil = 2;                ///< Number of vertices in the stencil
    static constexpr int kDims    = 3;                ///< Number of dimensions
    static constexpr int kDofs    = kDims * kStencil; ///< Total degrees of freedom

    /**
     * @brief Compute the point-point distance \f$ d = \|x - y\| \f$
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `6 x 1` stacked positions \f$ [x[0], x[1], x[2], y[0], y[1], y[2]]^T \f$
     * @return Distance value
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    TScalar Eval(TMatrixx const& x);

    /**
     * @brief Compute the gradient of point-point distance
     *
     * See python/geometry/distance.py
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `6 x 1` stacked positions \f$ [x[0], x[1], x[2], y[0], y[1], y[2]]^T \f$
     * @return `6 x 1` gradient vector \f$ \nabla d \f$
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    auto Gradient(TMatrixx const& x) -> math::linalg::mini::SVector<TScalar, kDofs>;

    /**
     * @brief Compute the Hessian of point-point distance
     *
     * See python/geometry/distance.py
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `6 x 1` stacked positions \f$ [x[0], x[1], x[2], y[0], y[1], y[2]]^T \f$
     * @return `6 x 6` Hessian matrix \f$ \nabla^2 d \f$
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    auto Hessian(TMatrixx const& x) -> math::linalg::mini::SMatrix<TScalar, kDofs, kDofs>;
};

/**
 * @brief Point-edge distance computation
 *
 * Computes the distance \f$ d = \frac{\|(b-a) \times (x-a)\|}{\|b-a\|} \f$ and its derivatives,
 * where \f$ (a, b) \f$ are the edge endpoints and \f$ x \f$ is a point.
 *
 * @tparam TScalar Floating point scalar type
 */
template <common::CFloatingPoint TScalar>
struct PointEdgeDistance
{
    using ScalarType              = TScalar;
    static constexpr int kStencil = 3;                ///< Number of vertices in the stencil
    static constexpr int kDims    = 3;                ///< Number of dimensions
    static constexpr int kDofs    = kDims * kStencil; ///< Total degrees of freedom

    /**
     * @brief Compute the point-edge distance
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `9 x 1` stacked positions \f$ [x[0], x[1], x[2], a[0], a[1], a[2], b[0], b[1],
     * b[2]]^T \f$
     * @return Distance value
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    TScalar Eval(TMatrixx const& x);

    /**
     * @brief Compute the gradient of point-edge distance
     *
     * See python/geometry/distance.py
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `9 x 1` stacked positions \f$ [x[0], x[1], x[2], a[0], a[1], a[2], b[0], b[1],
     * b[2]]^T \f$
     * @return `9 x 1` gradient vector \f$ \nabla d \f$
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    auto Gradient(TMatrixx const& x) -> math::linalg::mini::SVector<TScalar, kDofs>;

    /**
     * @brief Compute the Hessian of point-edge distance
     *
     * See python/geometry/distance.py
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `9 x 1` stacked positions \f$ [x[0], x[1], x[2], a[0], a[1], a[2], b[0], b[1],
     * b[2]]^T \f$
     * @return `9 x 9` Hessian matrix \f$ \nabla^2 d \f$
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    auto Hessian(TMatrixx const& x) -> math::linalg::mini::SMatrix<TScalar, kDofs, kDofs>;
};

/**
 * @brief Point-triangle signed distance computation
 *
 * Computes the signed distance \f$ d = \hat{n}^T (x - a) \f$ and its derivatives,
 * where \f$ \hat{n} = \frac{n}{\|n\|} \f$ is the normalized triangle normal,
 * \f$ n = (b - a) \times (c - a) \f$ is the triangle normal,
 * \f$ (a, b, c) \f$ are the triangle vertices in counter-clockwise order,
 * and \f$ x \f$ is a point.
 *
 * @tparam TScalar Floating point scalar type
 */
template <common::CFloatingPoint TScalar>
struct PointTriangleDistance
{
    using ScalarType              = TScalar;
    static constexpr int kStencil = 4;                ///< Number of vertices in the stencil
    static constexpr int kDims    = 3;                ///< Number of dimensions
    static constexpr int kDofs    = kDims * kStencil; ///< Total degrees of freedom

    /**
     * @brief Compute the point-triangle signed distance
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions
     *          \f$ [x[0], x[1], x[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]^T \f$
     * @return Signed distance value (positive above triangle, negative below)
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    TScalar Eval(TMatrixx const& x);

    /**
     * @brief Compute the gradient of point-triangle signed distance
     *
     * See python/geometry/distance.py
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions
     *          \f$ [x[0], x[1], x[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]^T \f$
     * @return `12 x 1` gradient vector \f$ \nabla d \f$
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    auto Gradient(TMatrixx const& x) -> math::linalg::mini::SVector<TScalar, kDofs>;

    /**
     * @brief Compute the Hessian of point-triangle signed distance
     *
     * See python/geometry/distance.py
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions
     *          \f$ [x[0], x[1], x[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]^T \f$
     * @return `12 x 12` Hessian matrix \f$ \nabla^2 d \f$
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    auto Hessian(TMatrixx const& x) -> math::linalg::mini::SMatrix<TScalar, kDofs, kDofs>;
};

/**
 * @brief Edge-edge signed distance computation with mollified norm
 *
 * Computes the signed distance \f$ d = \hat{n}^T (c - a) \f$ and its derivatives,
 * where \f$ \hat{n} = \frac{n}{\|n\|_\varepsilon} \f$ is the approximately normalized edge-edge
 * normal, \f$ n = (b - a) \times (d - c) \f$ is the edge-edge normal, \f$ (a, b) \f$ are the first
 * edge's endpoints, \f$ (c, d) \f$ are the second edge's endpoints, and \f$ \|n\|_\varepsilon =
 * \sqrt{\sum_i n_i^2 + \varepsilon^2} \f$ is the mollified norm for numerical stability when edges
 * are nearly parallel.
 *
 * @tparam TScalar Floating point scalar type
 */
template <common::CFloatingPoint TScalar>
struct EdgeEdgeDistance
{
    using ScalarType              = TScalar;
    static constexpr int kStencil = 4;                ///< Number of vertices in the stencil
    static constexpr int kDims    = 3;                ///< Number of dimensions
    static constexpr int kDofs    = kDims * kStencil; ///< Total degrees of freedom

    /**
     * @brief Compute the edge-edge signed distance with mollified norm
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions
     *          \f$ [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]]^T \f$
     * @param eps Mollification parameter \f$ \varepsilon \f$ for numerical stability
     * @return Signed distance value
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    TScalar Eval(TMatrixx const& x, ScalarType eps = ScalarType(1e-5));

    /**
     * @brief Compute the gradient of edge-edge signed distance
     *
     * See python/geometry/distance.py
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions
     *          \f$ [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]]^T \f$
     * @param eps Mollification parameter \f$ \varepsilon \f$ for numerical stability
     * @return `12 x 1` gradient vector \f$ \nabla d \f$
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    auto Gradient(TMatrixx const& x, ScalarType eps = ScalarType(1e-5))
        -> math::linalg::mini::SVector<TScalar, kDofs>;

    /**
     * @brief Compute the Hessian of edge-edge signed distance
     *
     * See python/geometry/distance.py
     *
     * @tparam TMatrixx Matrix type satisfying CMatrix concept
     * @param x `12 x 1` stacked positions
     *          \f$ [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]]^T \f$
     * @param eps Mollification parameter \f$ \varepsilon \f$ for numerical stability
     * @return `12 x 12` Hessian matrix \f$ \nabla^2 d \f$
     */
    template <math::linalg::mini::CMatrix TMatrixx>
    auto Hessian(TMatrixx const& x, ScalarType eps = ScalarType(1e-5))
        -> math::linalg::mini::SMatrix<TScalar, kDofs, kDofs>;
};

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline TScalar PointPointDistance<TScalar>::Eval(TMatrixx const& x_)
{
    auto x = x_.template Slice<3, 1>(0, 0);
    auto y = x_.template Slice<3, 1>(3, 0);
    return Norm(x - y);
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline auto PointPointDistance<TScalar>::Gradient(TMatrixx const& x_)
    -> math::linalg::mini::SVector<TScalar, kDofs>
{
    auto x = x_.template Slice<3, 1>(0, 0);
    auto y = x_.template Slice<3, 1>(3, 0);
    math::linalg::mini::SVector<TScalar, 6> g;
    auto gx                                   = g.template Slice<3, 1>(0, 0);
    auto gy                                   = g.template Slice<3, 1>(3, 0);
    math::linalg::mini::SVector<TScalar, 3> n = x - y;
    n /= Norm(n);
    gx = n;
    gy = -n;
    return g;
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline auto PointPointDistance<TScalar>::Hessian(TMatrixx const& x_)
    -> math::linalg::mini::SMatrix<TScalar, kDofs, kDofs>
{
    auto x                                    = x_.template Slice<3, 1>(0, 0);
    auto y                                    = x_.template Slice<3, 1>(3, 0);
    math::linalg::mini::SVector<TScalar, 3> n = x - y;
    TScalar d                                 = Norm(n);
    n /= d;
    math::linalg::mini::SMatrix<TScalar, kDofs, kDofs> H;
    math::linalg::mini::Identity<TScalar, 3, 3> I{};
    H.template Slice<3, 3>(0, 0) = (1 / d) * (I - n * n.Transpose());
    H.template Slice<3, 3>(3, 0) = -H.template Slice<3, 3>(0, 0);
    H.template Slice<3, 3>(0, 3) = -H.template Slice<3, 3>(0, 0);
    H.template Slice<3, 3>(3, 3) = H.template Slice<3, 3>(0, 0);
    return H;
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline TScalar PointEdgeDistance<TScalar>::Eval(TMatrixx const& x_)
{
    auto x = x_.template Slice<3, 1>(0, 0);
    auto a = x_.template Slice<3, 1>(3, 0);
    auto b = x_.template Slice<3, 1>(6, 0);
    using namespace std;
    return sqrt(DistanceQueries::PointLineSegment(x, a, b));
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline auto PointEdgeDistance<TScalar>::Gradient(TMatrixx const& x_)
    -> math::linalg::mini::SVector<TScalar, kDofs>
{
    auto x = x_.template Slice<3, 1>(0, 0);
    auto a = x_.template Slice<3, 1>(3, 0);
    auto b = x_.template Slice<3, 1>(6, 0);
    math::linalg::mini::SVector<TScalar, kDofs> g;
    auto gx = g.template Slice<3, 1>(0, 0);
    auto ga = g.template Slice<3, 1>(3, 0);
    auto gb = g.template Slice<3, 1>(6, 0);
    using namespace std;
    // NOTE: This is the actual point-edge distance function.
    // math::linalg::mini::SVector<TScalar, 3> ab = b - a;
    // math::linalg::mini::SVector<TScalar, 3> ax = x - a;
    // math::linalg::mini::SVector<TScalar, 3> n  = Cross(ab, ax);
    // TScalar nnorm                              = Norm(n);
    // n /= nnorm;
    // TScalar abnorm        = Norm(ab);
    // TScalar nnorm_abnorm2 = nnorm / (abnorm * abnorm);
    // gx                    = Cross(-ab, n);
    // ga                    = Cross(b - x, n) + nnorm_abnorm2 * ab;
    // gb                    = Cross(ax, n) - nnorm_abnorm2 * ab;
    // g *= (1 / abnorm);

    // NOTE: This is stabler, but translation-only.
    auto uv                                   = ClosestPointQueries::UvPointOnLineSegment(x, a, b);
    math::linalg::mini::SVector<TScalar, 3> n = x - (uv(0) * a + uv(1) * b);
    TScalar nnorm                             = Norm(n);
    TScalar invnnorm                          = 1 / nnorm;
    n *= invnnorm;
    gx = n;
    ga = -uv(0) * n;
    gb = -uv(1) * n;
    return g;
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline auto PointEdgeDistance<TScalar>::Hessian(TMatrixx const& x_)
    -> math::linalg::mini::SMatrix<TScalar, kDofs, kDofs>
{
    auto x = x_.template Slice<3, 1>(0, 0);
    auto a = x_.template Slice<3, 1>(3, 0);
    auto b = x_.template Slice<3, 1>(6, 0);
    using namespace std;
    TScalar a0   = a[1] - b[1];
    TScalar a1   = 2 * a[1];
    TScalar a2   = 2 * b[1];
    TScalar a3   = a1 - a2;
    TScalar a4   = (TScalar(1) / TScalar(2)) * a3;
    TScalar a5   = a[2] - b[2];
    TScalar a6   = 2 * a[2];
    TScalar a7   = 2 * b[2];
    TScalar a8   = (TScalar(1) / TScalar(2)) * a6 - TScalar(1) / TScalar(2) * a7;
    TScalar a9   = a5 * a8;
    TScalar a10  = a[0] - b[0];
    TScalar a11  = ((a0) * (a0)) + ((a10) * (a10)) + ((a5) * (a5));
    TScalar a12  = 1 / sqrt(a11);
    TScalar a13  = -a10;
    TScalar a14  = -x[1];
    TScalar a15  = a14 + a[1];
    TScalar a16  = -a15;
    TScalar a17  = a13 * a16;
    TScalar a18  = -x[0];
    TScalar a19  = a18 + a[0];
    TScalar a20  = -a0;
    TScalar a21  = -a19 * a20;
    TScalar a22  = a17 - a21;
    TScalar a23  = -x[2];
    TScalar a24  = a23 + a[2];
    TScalar a25  = a10 * a24;
    TScalar a26  = a19 * a5;
    TScalar a27  = a25 - a26;
    TScalar a28  = -a24;
    TScalar a29  = a20 * a28;
    TScalar a30  = -a5;
    TScalar a31  = a16 * a30;
    TScalar a32  = a29 - a31;
    TScalar a33  = ((a22) * (a22)) + ((a27) * (a27)) + ((a32) * (a32));
    TScalar a34  = sqrt(a33);
    TScalar a35  = TScalar(1) / a34;
    TScalar a36  = a12 * a35;
    TScalar a37  = a22 * a4 + a27 * a8;
    TScalar a38  = -a37;
    TScalar a39  = a12 / pow(a33, TScalar(3) / TScalar(2));
    TScalar a40  = a37 * a39;
    TScalar a41  = a13 * a36;
    TScalar a42  = 2 * a[0];
    TScalar a43  = 2 * b[0];
    TScalar a44  = -TScalar(1) / TScalar(2) * a42 + (TScalar(1) / TScalar(2)) * a43;
    TScalar a45  = a22 * a44 + a32 * a8;
    TScalar a46  = -a45;
    TScalar a47  = -a3;
    TScalar a48  = (TScalar(1) / TScalar(2)) * a32;
    TScalar a49  = a27 * a44 + a47 * a48;
    TScalar a50  = -a49;
    TScalar a51  = a14 + b[1];
    TScalar a52  = a23 + b[2];
    TScalar a53  = a52 * a8;
    TScalar a54  = pow(a11, -TScalar(3) / TScalar(2));
    TScalar a55  = a35 * a54;
    TScalar a56  = a37 * a55;
    TScalar a57  = a13 * a56;
    TScalar a58  = -2 * x[1];
    TScalar a59  = a2 + a58;
    TScalar a60  = (TScalar(1) / TScalar(2)) * a22;
    TScalar a61  = -2 * x[2];
    TScalar a62  = a61 + a7;
    TScalar a63  = (TScalar(1) / TScalar(2)) * a27;
    TScalar a64  = a59 * a60 + a62 * a63;
    TScalar a65  = -a64;
    TScalar a66  = -a18 - b[0];
    TScalar a67  = a20 * a56;
    TScalar a68  = -2 * x[0];
    TScalar a69  = -a43 - a68;
    TScalar a70  = a48 * a62 + a60 * a69;
    TScalar a71  = -a70;
    TScalar a72  = a30 * a56;
    TScalar a73  = -a59;
    TScalar a74  = a48 * a73 + a63 * a69;
    TScalar a75  = -a74;
    TScalar a76  = a28 * a8;
    TScalar a77  = a10 * a56;
    TScalar a78  = a1 + a58;
    TScalar a79  = -a78;
    TScalar a80  = -a6 - a61;
    TScalar a81  = a60 * a79 + a63 * a80;
    TScalar a82  = -a81;
    TScalar a83  = -a17 + a21;
    TScalar a84  = a0 * a56;
    TScalar a85  = a42 + a68;
    TScalar a86  = a48 * a80 + a60 * a85;
    TScalar a87  = -a86;
    TScalar a88  = -a25 + a26;
    TScalar a89  = a5 * a56;
    TScalar a90  = a48 * a78 + a63 * a85;
    TScalar a91  = -a90;
    TScalar a92  = a36 * a44;
    TScalar a93  = a39 * a45;
    TScalar a94  = a13 * a44;
    TScalar a95  = a45 * a55;
    TScalar a96  = a13 * a95;
    TScalar a97  = a44 * a66;
    TScalar a98  = a20 * a95;
    TScalar a99  = -a51;
    TScalar a100 = a30 * a95;
    TScalar a101 = a10 * a95;
    TScalar a102 = a19 * a44;
    TScalar a103 = a0 * a95;
    TScalar a104 = -a29 + a31;
    TScalar a105 = a5 * a95;
    TScalar a106 = a39 * a49;
    TScalar a107 = (TScalar(1) / TScalar(2)) * a47;
    TScalar a108 = a49 * a55;
    TScalar a109 = a108 * a13;
    TScalar a110 = a108 * a20;
    TScalar a111 = a108 * a30;
    TScalar a112 = a10 * a108;
    TScalar a113 = a0 * a108;
    TScalar a114 = a108 * a5;
    TScalar a115 = (TScalar(1) / TScalar(2)) * a59;
    TScalar a116 = (TScalar(1) / TScalar(2)) * a62;
    TScalar a117 = a116 * a5;
    TScalar a118 = a39 * a64;
    TScalar a119 = a34 * a54;
    TScalar a120 = -a119;
    TScalar a121 = 3 * a[0] - 3 * b[0];
    TScalar a122 = -a121;
    TScalar a123 = a34 / pow(a11, TScalar(5) / TScalar(2));
    TScalar a124 = a123 * a13;
    TScalar a125 = a116 * a52;
    TScalar a126 = a13 * a55;
    TScalar a127 = 3 * a[1] - 3 * b[1];
    TScalar a128 = -a127;
    TScalar a129 = a36 * a66;
    TScalar a130 = a55 * a64;
    TScalar a131 = a126 * a70 + a130 * a20;
    TScalar a132 = 3 * a[2] - 3 * b[2];
    TScalar a133 = -a132;
    TScalar a134 = a126 * a74 + a130 * a30;
    TScalar a135 = a116 * a28;
    TScalar a136 = a10 * a130 + a119 + a126 * a81;
    TScalar a137 = a0 * a130 + a126 * a86;
    TScalar a138 = a126 * a90 + a130 * a5;
    TScalar a139 = (TScalar(1) / TScalar(2)) * a69;
    TScalar a140 = a39 * a70;
    TScalar a141 = a13 * a139;
    TScalar a142 = a123 * a20;
    TScalar a143 = a139 * a36;
    TScalar a144 = a139 * a66;
    TScalar a145 = a20 * a55;
    TScalar a146 = a55 * a70;
    TScalar a147 = a145 * a74 + a146 * a30;
    TScalar a148 = a10 * a146 + a145 * a81;
    TScalar a149 = a139 * a19;
    TScalar a150 = a0 * a146 + a119 + a145 * a86;
    TScalar a151 = a145 * a90 + a146 * a5;
    TScalar a152 = a39 * a74;
    TScalar a153 = (TScalar(1) / TScalar(2)) * a73;
    TScalar a154 = a123 * a30;
    TScalar a155 = a55 * a74;
    TScalar a156 = a30 * a55;
    TScalar a157 = a10 * a155 + a156 * a81;
    TScalar a158 = a0 * a155 + a156 * a86;
    TScalar a159 = a119 + a155 * a5 + a156 * a90;
    TScalar a160 = (TScalar(1) / TScalar(2)) * a79;
    TScalar a161 = (TScalar(1) / TScalar(2)) * a80;
    TScalar a162 = a161 * a5;
    TScalar a163 = a39 * a81;
    TScalar a164 = a161 * a52;
    TScalar a165 = a10 * a123;
    TScalar a166 = a161 * a28;
    TScalar a167 = a10 * a55;
    TScalar a168 = a19 * a36;
    TScalar a169 = a55 * a81;
    TScalar a170 = a0 * a169 + a167 * a86;
    TScalar a171 = a167 * a90 + a169 * a5;
    TScalar a172 = (TScalar(1) / TScalar(2)) * a85;
    TScalar a173 = a39 * a86;
    TScalar a174 = a13 * a172;
    TScalar a175 = a0 * a123;
    TScalar a176 = a172 * a66;
    TScalar a177 = a172 * a36;
    TScalar a178 = a172 * a19;
    TScalar a179 = a0 * a55;
    TScalar a180 = a5 * a55;
    TScalar a181 = a179 * a90 + a180 * a86;
    TScalar a182 = a39 * a90;
    TScalar a183 = (TScalar(1) / TScalar(2)) * a78;
    TScalar a184 = a123 * a5;
    math::linalg::mini::SMatrix<TScalar, kDofs, kDofs> hess_d;
    hess_d[0]  = a36 * (a0 * a4 + a9) + a38 * a40;
    hess_d[1]  = a4 * a41 + a40 * a46;
    hess_d[2]  = a40 * a50 + a41 * a8;
    hess_d[3]  = a36 * (a4 * a51 + a53) + a40 * a65 + a57;
    hess_d[4]  = a36 * (a22 + a4 * a66) + a40 * a71 + a67;
    hess_d[5]  = a36 * (a27 + a66 * a8) + a40 * a75 + a72;
    hess_d[6]  = a36 * (a16 * a4 + a76) + a40 * a82 + a77;
    hess_d[7]  = a36 * (a19 * a4 + a83) + a40 * a87 + a84;
    hess_d[8]  = a36 * (a19 * a8 + a88) + a40 * a91 + a89;
    hess_d[9]  = a0 * a92 + a38 * a93;
    hess_d[10] = a36 * (a9 + a94) + a46 * a93;
    hess_d[11] = a20 * a36 * a8 + a50 * a93;
    hess_d[12] = a36 * (a44 * a51 + a83) + a65 * a93 + a96;
    hess_d[13] = a36 * (a53 + a97) + a71 * a93 + a98;
    hess_d[14] = a100 + a36 * (a32 + a8 * a99) + a75 * a93;
    hess_d[15] = a101 + a36 * (a16 * a44 + a22) + a82 * a93;
    hess_d[16] = a103 + a36 * (a102 + a76) + a87 * a93;
    hess_d[17] = a105 + a36 * (a104 + a15 * a8) + a91 * a93;
    hess_d[18] = a106 * a38 + a5 * a92;
    hess_d[19] = a106 * a46 + a107 * a36 * a5;
    hess_d[20] = a106 * a50 + a36 * (a107 * a20 + a94);
    hess_d[21] = a106 * a65 + a109 + a36 * (a44 * a52 + a88);
    hess_d[22] = a106 * a71 + a110 + a36 * (a104 + a107 * a52);
    hess_d[23] = a106 * a75 + a111 + a36 * (a107 * a99 + a97);
    hess_d[24] = a106 * a82 + a112 + a36 * (a27 + a28 * a44);
    hess_d[25] = a106 * a87 + a113 + a36 * (a107 * a28 + a32);
    hess_d[26] = a106 * a91 + a114 + a36 * (a102 + a107 * a15);
    hess_d[27] = a118 * a38 + a36 * (a0 * a115 + a117) + a57;
    hess_d[28] = a118 * a46 + a36 * (a115 * a13 + a83) + a96;
    hess_d[29] = a109 + a118 * a50 + a36 * (a116 * a13 + a88);
    hess_d[30] = a118 * a65 + a120 + a122 * a124 + 2 * a126 * a64 + a36 * (a115 * a51 + a125);
    hess_d[31] = a115 * a129 + a118 * a71 + a124 * a128 + a131;
    hess_d[32] = a116 * a129 + a118 * a75 + a124 * a133 + a134;
    hess_d[33] = a118 * a82 + a121 * a124 + a136 + a36 * (a115 * a16 + a135);
    hess_d[34] = a118 * a87 + a124 * a127 + a137 + a36 * (a115 * a19 + a22);
    hess_d[35] = a118 * a91 + a124 * a132 + a138 + a36 * (a116 * a19 + a27);
    hess_d[36] = a140 * a38 + a36 * (a0 * a139 + a22) + a67;
    hess_d[37] = a140 * a46 + a36 * (a117 + a141) + a98;
    hess_d[38] = a110 + a140 * a50 + a36 * (a104 + a116 * a20);
    hess_d[39] = a122 * a142 + a131 + a140 * a65 + a143 * a51;
    hess_d[40] = a120 + a128 * a142 + a140 * a71 + 2 * a145 * a70 + a36 * (a125 + a144);
    hess_d[41] = a116 * a36 * a99 + a133 * a142 + a140 * a75 + a147;
    hess_d[42] = a121 * a142 + a140 * a82 + a148 + a36 * (a139 * a16 + a83);
    hess_d[43] = a127 * a142 + a140 * a87 + a150 + a36 * (a135 + a149);
    hess_d[44] = a132 * a142 + a140 * a91 + a151 + a36 * (a116 * a15 + a32);
    hess_d[45] = a152 * a38 + a36 * (a139 * a5 + a27) + a72;
    hess_d[46] = a100 + a152 * a46 + a36 * (a153 * a5 + a32);
    hess_d[47] = a111 + a152 * a50 + a36 * (a141 + a153 * a20);
    hess_d[48] = a122 * a154 + a134 + a143 * a52 + a152 * a65;
    hess_d[49] = a128 * a154 + a147 + a152 * a71 + a153 * a36 * a52;
    hess_d[50] = a120 + a133 * a154 + a152 * a75 + 2 * a155 * a30 + a36 * (a144 + a153 * a99);
    hess_d[51] = a121 * a154 + a152 * a82 + a157 + a36 * (a139 * a28 + a88);
    hess_d[52] = a127 * a154 + a152 * a87 + a158 + a36 * (a104 + a153 * a28);
    hess_d[53] = a132 * a154 + a152 * a91 + a159 + a36 * (a149 + a15 * a153);
    hess_d[54] = a163 * a38 + a36 * (a0 * a160 + a162) + a77;
    hess_d[55] = a101 + a163 * a46 + a36 * (a13 * a160 + a22);
    hess_d[56] = a112 + a163 * a50 + a36 * (a13 * a161 + a27);
    hess_d[57] = a122 * a165 + a136 + a163 * a65 + a36 * (a160 * a51 + a164);
    hess_d[58] = a128 * a165 + a148 + a163 * a71 + a36 * (a160 * a66 + a83);
    hess_d[59] = a133 * a165 + a157 + a163 * a75 + a36 * (a161 * a66 + a88);
    hess_d[60] = a120 + a121 * a165 + a163 * a82 + 2 * a167 * a81 + a36 * (a16 * a160 + a166);
    hess_d[61] = a127 * a165 + a160 * a168 + a163 * a87 + a170;
    hess_d[62] = a132 * a165 + a161 * a168 + a163 * a91 + a171;
    hess_d[63] = a173 * a38 + a36 * (a0 * a172 + a83) + a84;
    hess_d[64] = a103 + a173 * a46 + a36 * (a162 + a174);
    hess_d[65] = a113 + a173 * a50 + a36 * (a161 * a20 + a32);
    hess_d[66] = a122 * a175 + a137 + a173 * a65 + a36 * (a172 * a51 + a22);
    hess_d[67] = a128 * a175 + a150 + a173 * a71 + a36 * (a164 + a176);
    hess_d[68] = a133 * a175 + a158 + a173 * a75 + a36 * (a104 + a161 * a99);
    hess_d[69] = a121 * a175 + a16 * a177 + a170 + a173 * a82;
    hess_d[70] = a120 + a127 * a175 + a173 * a87 + 2 * a179 * a86 + a36 * (a166 + a178);
    hess_d[71] = a132 * a175 + a15 * a161 * a36 + a173 * a91 + a181;
    hess_d[72] = a182 * a38 + a36 * (a172 * a5 + a88) + a89;
    hess_d[73] = a105 + a182 * a46 + a36 * (a104 + a183 * a5);
    hess_d[74] = a114 + a182 * a50 + a36 * (a174 + a183 * a20);
    hess_d[75] = a122 * a184 + a138 + a182 * a65 + a36 * (a172 * a52 + a27);
    hess_d[76] = a128 * a184 + a151 + a182 * a71 + a36 * (a183 * a52 + a32);
    hess_d[77] = a133 * a184 + a159 + a182 * a75 + a36 * (a176 + a183 * a99);
    hess_d[78] = a121 * a184 + a171 + a177 * a28 + a182 * a82;
    hess_d[79] = a127 * a184 + a181 + a182 * a87 + a183 * a28 * a36;
    hess_d[80] = a120 + a132 * a184 + 2 * a180 * a90 + a182 * a91 + a36 * (a15 * a183 + a178);
    return hess_d;
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline TScalar PointTriangleDistance<TScalar>::Eval(TMatrixx const& x_)
{
    auto x = x_.template Slice<3, 1>(0, 0);
    auto a = x_.template Slice<3, 1>(3, 0);
    auto b = x_.template Slice<3, 1>(6, 0);
    auto c = x_.template Slice<3, 1>(9, 0);
    using namespace std;
    return sqrt(DistanceQueries::PointTriangle(x, a, b, c));
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline auto PointTriangleDistance<TScalar>::Gradient(TMatrixx const& x_)
    -> math::linalg::mini::SVector<TScalar, kDofs>
{
    auto x = x_.template Slice<3, 1>(0, 0);
    auto a = x_.template Slice<3, 1>(3, 0);
    auto b = x_.template Slice<3, 1>(6, 0);
    auto c = x_.template Slice<3, 1>(9, 0);
    math::linalg::mini::SVector<TScalar, kDofs> g;
    auto gx = g.template Slice<3, 1>(0, 0);
    auto ga = g.template Slice<3, 1>(3, 0);
    auto gb = g.template Slice<3, 1>(6, 0);
    auto gc = g.template Slice<3, 1>(9, 0);
    using namespace std;
    // NOTE: This is the actual point-triangle distance function, but it's not accurate away from
    // the triangle.
    // math::linalg::mini::SVector<TScalar, 3> ab = b - a;
    // math::linalg::mini::SVector<TScalar, 3> ac = c - a;
    // math::linalg::mini::SVector<TScalar, 3> ax = x - a;
    // math::linalg::mini::SVector<TScalar, 3> n  = Cross(ab, ac);
    // TScalar nnorm                              = Norm(n);
    // TScalar nnorminv                           = 1 / nnorm;
    // n *= nnorminv;
    // math::linalg::mini::Identity<TScalar, 3, 3> I;
    // math::linalg::mini::SVector<TScalar, 3> Pnax = ax - Dot(ax, n) * n;
    // auto uvw = ClosestPointQueries::UvwPointInTriangle(x, a, b, c);
    // gx       = n;
    // ga       = nnorminv * (Cross(b - c, Pnax)) - uvw(0) * n;
    // gb       = nnorminv * (Cross(ac, Pnax)) - uvw(1) * n;
    // gc       = nnorminv * (Cross(-ab, Pnax)) - uvw(2) * n;

    // NOTE: Use a translation-only point-triangle distance query
    auto uvw = ClosestPointQueries::UvwPointInTriangle(x, a, b, c);
    math::linalg::mini::SVector<TScalar, 3> xc = uvw(0) * a + uvw(1) * b + uvw(2) * c;
    math::linalg::mini::SVector<TScalar, 3> n  = x - xc;
    TScalar nnorm                              = Norm(n);
    TScalar nnorminv                           = 1 / nnorm;
    n *= nnorminv;
    gx = n;
    ga = -uvw(0) * n;
    gb = -uvw(1) * n;
    gc = -uvw(2) * n;
    return g;
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline auto PointTriangleDistance<TScalar>::Hessian(TMatrixx const& x_)
    -> math::linalg::mini::SMatrix<TScalar, kDofs, kDofs>
{
    auto x = x_.template Slice<3, 1>(0, 0);
    auto a = x_.template Slice<3, 1>(3, 0);
    auto b = x_.template Slice<3, 1>(6, 0);
    auto c = x_.template Slice<3, 1>(9, 0);
    using namespace std;
    TScalar a0   = 2 * b[1];
    TScalar a1   = -2 * c[1];
    TScalar a2   = a0 + a1;
    TScalar a3   = a[0] - b[0];
    TScalar a4   = -a3;
    TScalar a5   = -c[1];
    TScalar a6   = a5 + a[1];
    TScalar a7   = -a6;
    TScalar a8   = a4 * a7;
    TScalar a9   = -c[0];
    TScalar a10  = a9 + a[0];
    TScalar a11  = -a10;
    TScalar a12  = a[1] - b[1];
    TScalar a13  = -a12;
    TScalar a14  = a11 * a13;
    TScalar a15  = -a14 + a8;
    TScalar a16  = (TScalar(1) / TScalar(2)) * a15;
    TScalar a17  = a16 * a2;
    TScalar a18  = -c[2];
    TScalar a19  = a18 + a[2];
    TScalar a20  = a19 * a3;
    TScalar a21  = a[2] - b[2];
    TScalar a22  = a10 * a21;
    TScalar a23  = a20 - a22;
    TScalar a24  = 2 * b[2];
    TScalar a25  = -2 * c[2];
    TScalar a26  = a24 + a25;
    TScalar a27  = (TScalar(1) / TScalar(2)) * a26;
    TScalar a28  = -a17 - a23 * a27;
    TScalar a29  = -a19;
    TScalar a30  = a13 * a29;
    TScalar a31  = -a21;
    TScalar a32  = a31 * a7;
    TScalar a33  = a30 - a32;
    TScalar a34  = ((a15) * (a15)) + ((a23) * (a23)) + ((a33) * (a33));
    TScalar a35  = pow(a34, -TScalar(3) / TScalar(2));
    TScalar a36  = a33 * a35;
    TScalar a37  = a28 * a36;
    TScalar a38  = a18 + b[2];
    TScalar a39  = 1 / sqrt(a34);
    TScalar a40  = 2 * b[0];
    TScalar a41  = -2 * c[0];
    TScalar a42  = -a40 - a41;
    TScalar a43  = a16 * a42;
    TScalar a44  = -a27 * a33 - a43;
    TScalar a45  = a36 * a44 + a38 * a39;
    TScalar a46  = a5 + b[1];
    TScalar a47  = -a46;
    TScalar a48  = (TScalar(1) / TScalar(2)) * a23;
    TScalar a49  = -a2;
    TScalar a50  = (TScalar(1) / TScalar(2)) * a33;
    TScalar a51  = -a42 * a48 - a49 * a50;
    TScalar a52  = a36 * a51 + a39 * a47;
    TScalar a53  = 2 * a[1];
    TScalar a54  = a1 + a53;
    TScalar a55  = -a54;
    TScalar a56  = a16 * a55;
    TScalar a57  = 2 * a[2];
    TScalar a58  = -a25 - a57;
    TScalar a59  = -a48 * a58 - a56;
    TScalar a60  = a36 * a59;
    TScalar a61  = a29 * a39;
    TScalar a62  = 2 * a[0];
    TScalar a63  = a41 + a62;
    TScalar a64  = a16 * a63;
    TScalar a65  = -a50 * a58 - a64;
    TScalar a66  = a36 * a65;
    TScalar a67  = a61 + a66;
    TScalar a68  = a39 * a6;
    TScalar a69  = -a48 * a63 - a50 * a54;
    TScalar a70  = a36 * a69;
    TScalar a71  = a68 + a70;
    TScalar a72  = -a0 + a53;
    TScalar a73  = a16 * a72;
    TScalar a74  = -a24 + a57;
    TScalar a75  = -a48 * a74 - a73;
    TScalar a76  = a36 * a75;
    TScalar a77  = a21 * a39;
    TScalar a78  = a40 - a62;
    TScalar a79  = a16 * a78;
    TScalar a80  = -a50 * a74 - a79;
    TScalar a81  = a36 * a80;
    TScalar a82  = a77 + a81;
    TScalar a83  = a13 * a39;
    TScalar a84  = -a72;
    TScalar a85  = -a48 * a78 - a50 * a84;
    TScalar a86  = a36 * a85;
    TScalar a87  = a83 + a86;
    TScalar a88  = -a38;
    TScalar a89  = a11 * a31 - a29 * a4;
    TScalar a90  = a35 * a89;
    TScalar a91  = a28 * a90 + a39 * a88;
    TScalar a92  = a44 * a90;
    TScalar a93  = a9 + b[0];
    TScalar a94  = a39 * a93 + a51 * a90;
    TScalar a95  = a19 * a39;
    TScalar a96  = a59 * a90;
    TScalar a97  = a95 + a96;
    TScalar a98  = a65 * a90;
    TScalar a99  = a11 * a39;
    TScalar a100 = a69 * a90;
    TScalar a101 = a100 + a99;
    TScalar a102 = a31 * a39;
    TScalar a103 = a75 * a90;
    TScalar a104 = a102 + a103;
    TScalar a105 = a80 * a90;
    TScalar a106 = a3 * a39;
    TScalar a107 = a85 * a90;
    TScalar a108 = a106 + a107;
    TScalar a109 = a15 * a35;
    TScalar a110 = a109 * a28 + a39 * a46;
    TScalar a111 = -a93;
    TScalar a112 = a109 * a44 + a111 * a39;
    TScalar a113 = a109 * a51;
    TScalar a114 = a39 * a7;
    TScalar a115 = a109 * a59;
    TScalar a116 = a114 + a115;
    TScalar a117 = a10 * a39;
    TScalar a118 = a109 * a65;
    TScalar a119 = a117 + a118;
    TScalar a120 = a109 * a69;
    TScalar a121 = a12 * a39;
    TScalar a122 = a109 * a75;
    TScalar a123 = a121 + a122;
    TScalar a124 = a39 * a4;
    TScalar a125 = a109 * a80;
    TScalar a126 = a124 + a125;
    TScalar a127 = a109 * a85;
    TScalar a128 = (TScalar(1) / TScalar(2)) * a2;
    TScalar a129 = a27 * a38;
    TScalar a130 = -a128 * a46 - a129;
    TScalar a131 = -a[0] + x[0];
    TScalar a132 = a131 * a36;
    TScalar a133 = -a[1] + x[1];
    TScalar a134 = a133 * a90;
    TScalar a135 = -a[2] + x[2];
    TScalar a136 = a109 * a135;
    TScalar a137 = a28 * a35;
    TScalar a138 = a133 * a88;
    TScalar a139 = a135 * a35;
    TScalar a140 = a139 * a46;
    TScalar a141 = (TScalar(3) / TScalar(2)) * a15;
    TScalar a142 = (TScalar(3) / TScalar(2)) * a26;
    TScalar a143 = -a141 * a2 - a142 * a23;
    TScalar a144 = pow(a34, -TScalar(5) / TScalar(2));
    TScalar a145 = a144 * a28;
    TScalar a146 = a143 * a145;
    TScalar a147 = a131 * a33;
    TScalar a148 = a133 * a89;
    TScalar a149 = a135 * a15;
    TScalar a150 = a111 * a128;
    TScalar a151 = a111 * a139;
    TScalar a152 = -a141 * a42 - a142 * a33;
    TScalar a153 = -a111 * a135 * a28 * a35 - a131 * a28 * a35 * a38 - a133 * a35 * a44 * a88 -
                   a135 * a35 * a44 * a46 + a45 + a91;
    TScalar a154 = a111 * a27;
    TScalar a155 = a16 * a26;
    TScalar a156 = (TScalar(3) / TScalar(2)) * a23;
    TScalar a157 = (TScalar(3) / TScalar(2)) * a33;
    TScalar a158 = -a156 * a42 - a157 * a49;
    TScalar a159 = a110 - a131 * a28 * a35 * a47 - a133 * a28 * a35 * a93 - a133 * a35 * a51 * a88 -
                   a135 * a35 * a46 * a51 + a52;
    TScalar a160 = a27 * a29;
    TScalar a161 = -a128 * a7 - a160;
    TScalar a162 = -a141 * a55 - a156 * a58;
    TScalar a163 = a145 * a162;
    TScalar a164 = a133 * a137;
    TScalar a165 = a139 * a28;
    TScalar a166 = a138 * a35;
    TScalar a167 = a140 * a59 + a164 * a19 + a165 * a7 + a166 * a59 - a60;
    TScalar a168 = -a10 * a128 - a15;
    TScalar a169 = -a141 * a63 - a157 * a58;
    TScalar a170 = a145 * a169;
    TScalar a171 = a135 * a39;
    TScalar a172 = a131 * a137;
    TScalar a173 = a10 * a165 + a140 * a65 + a166 * a65 + a171 + a172 * a29 - a61 - a66;
    TScalar a174 = -a10 * a27 - a23;
    TScalar a175 = -a156 * a63 - a157 * a54;
    TScalar a176 = a145 * a175;
    TScalar a177 = a133 * a39;
    TScalar a178 = -a177;
    TScalar a179 = a11 * a164 + a140 * a69 + a166 * a69 + a172 * a6 + a178 - a68 - a70;
    TScalar a180 = a21 * a27;
    TScalar a181 = -a12 * a128 - a180;
    TScalar a182 = -a141 * a72 - a156 * a74;
    TScalar a183 = a145 * a182;
    TScalar a184 = a12 * a165 + a140 * a75 + a164 * a31 + a166 * a75 - a76;
    TScalar a185 = a14 - a8;
    TScalar a186 = -a128 * a4 - a185;
    TScalar a187 = -a141 * a78 - a157 * a74;
    TScalar a188 = a145 * a187;
    TScalar a189 = -a171;
    TScalar a190 = a140 * a80 + a165 * a4 + a166 * a80 + a172 * a21 + a189 - a77 - a81;
    TScalar a191 = -a20 + a22;
    TScalar a192 = -a191 - a27 * a4;
    TScalar a193 = -a156 * a78 - a157 * a84;
    TScalar a194 = a145 * a193;
    TScalar a195 = a13 * a172 + a140 * a85 + a164 * a3 + a166 * a85 + a177 - a83 - a86;
    TScalar a196 = (TScalar(1) / TScalar(2)) * a42;
    TScalar a197 = a196 * a46;
    TScalar a198 = a111 * a196;
    TScalar a199 = -a129 - a198;
    TScalar a200 = a35 * a44;
    TScalar a201 = a131 * a38;
    TScalar a202 = a144 * a44;
    TScalar a203 = a152 * a202;
    TScalar a204 = a27 * a47;
    TScalar a205 = -a111 * a135 * a35 * a51 + a112 - a131 * a35 * a38 * a51 -
                   a131 * a35 * a44 * a47 - a133 * a35 * a44 * a93 + a94;
    TScalar a206 = -a185 - a196 * a7;
    TScalar a207 = a162 * a202;
    TScalar a208 = a133 * a200;
    TScalar a209 = a201 * a35;
    TScalar a210 = a139 * a44;
    TScalar a211 = a151 * a59 + a189 + a19 * a208 + a209 * a59 + a210 * a7 - a95 - a96;
    TScalar a212 = a10 * a196;
    TScalar a213 = -a160 - a212;
    TScalar a214 = a169 * a202;
    TScalar a215 = a131 * a200;
    TScalar a216 = a10 * a210 + a151 * a65 + a209 * a65 + a215 * a29 - a98;
    TScalar a217 = -a27 * a6 - a33;
    TScalar a218 = a175 * a202;
    TScalar a219 = a131 * a39;
    TScalar a220 = -a100 + a11 * a208 + a151 * a69 + a209 * a69 + a215 * a6 + a219 - a99;
    TScalar a221 = -a12 * a196 - a15;
    TScalar a222 = a182 * a202;
    TScalar a223 = -a102 - a103 + a12 * a210 + a151 * a75 + a171 + a208 * a31 + a209 * a75;
    TScalar a224 = a196 * a4;
    TScalar a225 = -a180 - a224;
    TScalar a226 = a187 * a202;
    TScalar a227 = -a105 + a151 * a80 + a209 * a80 + a21 * a215 + a210 * a4;
    TScalar a228 = -a30 + a32;
    TScalar a229 = -a13 * a27 - a228;
    TScalar a230 = a193 * a202;
    TScalar a231 = -a219;
    TScalar a232 = -a106 - a107 + a13 * a215 + a151 * a85 + a208 * a3 + a209 * a85 + a231;
    TScalar a233 = a196 * a38;
    TScalar a234 = a139 * a38;
    TScalar a235 = (TScalar(1) / TScalar(2)) * a49;
    TScalar a236 = a235 * a38;
    TScalar a237 = -a198 - a235 * a47;
    TScalar a238 = a35 * a51;
    TScalar a239 = a131 * a47;
    TScalar a240 = a133 * a238;
    TScalar a241 = a144 * a51;
    TScalar a242 = a158 * a241;
    TScalar a243 = -a191 - a196 * a29;
    TScalar a244 = a162 * a241;
    TScalar a245 = a35 * a59;
    TScalar a246 = a133 * a93;
    TScalar a247 = a139 * a51;
    TScalar a248 = -a114 - a115 + a177 + a19 * a240 + a239 * a245 + a245 * a246 + a247 * a7;
    TScalar a249 = -a228 - a235 * a29;
    TScalar a250 = a169 * a241;
    TScalar a251 = a35 * a65;
    TScalar a252 = a131 * a238;
    TScalar a253 = a10 * a247 - a117 - a118 + a231 + a239 * a251 + a246 * a251 + a252 * a29;
    TScalar a254 = -a212 - a235 * a6;
    TScalar a255 = a175 * a241;
    TScalar a256 = a35 * a69;
    TScalar a257 = a11 * a240 - a120 + a239 * a256 + a246 * a256 + a252 * a6;
    TScalar a258 = -a196 * a21 - a23;
    TScalar a259 = a182 * a241;
    TScalar a260 = a35 * a75;
    TScalar a261 = a12 * a247 - a121 - a122 + a178 + a239 * a260 + a240 * a31 + a246 * a260;
    TScalar a262 = -a21 * a235 - a33;
    TScalar a263 = a187 * a241;
    TScalar a264 = a35 * a80;
    TScalar a265 = -a124 - a125 + a21 * a252 + a219 + a239 * a264 + a246 * a264 + a247 * a4;
    TScalar a266 = -a13 * a235 - a224;
    TScalar a267 = a193 * a241;
    TScalar a268 = a35 * a85;
    TScalar a269 = -a127 + a13 * a252 + a239 * a268 + a240 * a3 + a246 * a268;
    TScalar a270 = (TScalar(1) / TScalar(2)) * a55;
    TScalar a271 = (TScalar(1) / TScalar(2)) * a58;
    TScalar a272 = a271 * a38;
    TScalar a273 = -a270 * a46 - a272;
    TScalar a274 = a144 * a59;
    TScalar a275 = a143 * a274;
    TScalar a276 = -a111 * a270 - a185;
    TScalar a277 = a152 * a274;
    TScalar a278 = -a111 * a271 - a191;
    TScalar a279 = a158 * a274;
    TScalar a280 = a271 * a29;
    TScalar a281 = -a270 * a7 - a280;
    TScalar a282 = a139 * a59;
    TScalar a283 = a133 * a19;
    TScalar a284 = a162 * a274;
    TScalar a285 = a169 * a274;
    TScalar a286 = a10 * a270;
    TScalar a287 = a10 * a139;
    TScalar a288 = a139 * a7;
    TScalar a289 = a131 * a245;
    TScalar a290 = a10 * a282 + a251 * a283 + a288 * a65 + a289 * a29;
    TScalar a291 = a175 * a274;
    TScalar a292 = a10 * a271;
    TScalar a293 = a16 * a58;
    TScalar a294 = a133 * a245;
    TScalar a295 = a11 * a294 + a256 * a283 + a288 * a69 + a289 * a6;
    TScalar a296 = a21 * a271;
    TScalar a297 = -a12 * a270 - a296;
    TScalar a298 = a182 * a274;
    TScalar a299 = a12 * a282 + a260 * a283 + a288 * a75 + a294 * a31;
    TScalar a300 = -a15 - a270 * a4;
    TScalar a301 = a187 * a274;
    TScalar a302 = a171 + a21 * a289 + a264 * a283 + a282 * a4 + a288 * a80;
    TScalar a303 = -a23 - a271 * a4;
    TScalar a304 = a193 * a274;
    TScalar a305 = a13 * a289 + a178 + a268 * a283 + a288 * a85 + a294 * a3;
    TScalar a306 = (TScalar(1) / TScalar(2)) * a63;
    TScalar a307 = -a15 - a306 * a46;
    TScalar a308 = a144 * a65;
    TScalar a309 = a143 * a308;
    TScalar a310 = a111 * a306;
    TScalar a311 = -a272 - a310;
    TScalar a312 = a152 * a308;
    TScalar a313 = -a228 - a271 * a47;
    TScalar a314 = a158 * a308;
    TScalar a315 = a162 * a308;
    TScalar a316 = a306 * a7;
    TScalar a317 = a10 * a306;
    TScalar a318 = -a280 - a317;
    TScalar a319 = a131 * a251;
    TScalar a320 = a169 * a308;
    TScalar a321 = a175 * a308;
    TScalar a322 = a271 * a6;
    TScalar a323 = a133 * a251;
    TScalar a324 = a131 * a29;
    TScalar a325 = a11 * a323 + a256 * a324 + a287 * a69 + a319 * a6;
    TScalar a326 = -a12 * a306 - a185;
    TScalar a327 = a182 * a308;
    TScalar a328 = a139 * a65;
    TScalar a329 = a12 * a328 + a189 + a260 * a324 + a287 * a75 + a31 * a323;
    TScalar a330 = a306 * a4;
    TScalar a331 = -a296 - a330;
    TScalar a332 = a187 * a308;
    TScalar a333 = a21 * a319 + a264 * a324 + a287 * a80 + a328 * a4;
    TScalar a334 = -a13 * a271 - a33;
    TScalar a335 = a193 * a308;
    TScalar a336 = a13 * a319 + a219 + a268 * a324 + a287 * a85 + a3 * a323;
    TScalar a337 = -a23 - a306 * a38;
    TScalar a338 = a144 * a69;
    TScalar a339 = a143 * a338;
    TScalar a340 = (TScalar(1) / TScalar(2)) * a54;
    TScalar a341 = -a33 - a340 * a38;
    TScalar a342 = a152 * a338;
    TScalar a343 = -a310 - a340 * a47;
    TScalar a344 = a158 * a338;
    TScalar a345 = a162 * a338;
    TScalar a346 = a29 * a306;
    TScalar a347 = a139 * a29;
    TScalar a348 = a169 * a338;
    TScalar a349 = a29 * a340;
    TScalar a350 = -a317 - a340 * a6;
    TScalar a351 = a133 * a256;
    TScalar a352 = a131 * a6;
    TScalar a353 = a175 * a338;
    TScalar a354 = -a191 - a21 * a306;
    TScalar a355 = a182 * a338;
    TScalar a356 = a139 * a69;
    TScalar a357 = a11 * a133;
    TScalar a358 = a12 * a356 + a177 + a260 * a352 + a260 * a357 + a31 * a351;
    TScalar a359 = -a21 * a340 - a228;
    TScalar a360 = a187 * a338;
    TScalar a361 = a131 * a256;
    TScalar a362 = a21 * a361 + a231 + a264 * a352 + a264 * a357 + a356 * a4;
    TScalar a363 = -a13 * a340 - a330;
    TScalar a364 = a193 * a338;
    TScalar a365 = a13 * a361 + a268 * a352 + a268 * a357 + a3 * a351;
    TScalar a366 = (TScalar(1) / TScalar(2)) * a72;
    TScalar a367 = (TScalar(1) / TScalar(2)) * a74;
    TScalar a368 = a367 * a38;
    TScalar a369 = -a366 * a46 - a368;
    TScalar a370 = a144 * a75;
    TScalar a371 = a143 * a370;
    TScalar a372 = -a111 * a366 - a15;
    TScalar a373 = a152 * a370;
    TScalar a374 = -a111 * a367 - a23;
    TScalar a375 = a158 * a370;
    TScalar a376 = a29 * a367;
    TScalar a377 = -a366 * a7 - a376;
    TScalar a378 = a162 * a370;
    TScalar a379 = -a10 * a366 - a185;
    TScalar a380 = a169 * a370;
    TScalar a381 = -a10 * a367 - a191;
    TScalar a382 = a175 * a370;
    TScalar a383 = a21 * a367;
    TScalar a384 = -a12 * a366 - a383;
    TScalar a385 = a133 * a31;
    TScalar a386 = a12 * a139;
    TScalar a387 = a182 * a370;
    TScalar a388 = a187 * a370;
    TScalar a389 = a366 * a4;
    TScalar a390 = a139 * a4;
    TScalar a391 = a131 * a260;
    TScalar a392 = a21 * a391 + a264 * a385 + a386 * a80 + a390 * a75;
    TScalar a393 = a193 * a370;
    TScalar a394 = a367 * a4;
    TScalar a395 = a16 * a74;
    TScalar a396 = a133 * a3;
    TScalar a397 = a13 * a391 + a260 * a396 + a268 * a385 + a386 * a85;
    TScalar a398 = (TScalar(1) / TScalar(2)) * a78;
    TScalar a399 = -a185 - a398 * a46;
    TScalar a400 = a144 * a80;
    TScalar a401 = a143 * a400;
    TScalar a402 = a111 * a398;
    TScalar a403 = -a368 - a402;
    TScalar a404 = a152 * a400;
    TScalar a405 = -a33 - a367 * a47;
    TScalar a406 = a158 * a400;
    TScalar a407 = -a15 - a398 * a7;
    TScalar a408 = a162 * a400;
    TScalar a409 = a10 * a398;
    TScalar a410 = -a376 - a409;
    TScalar a411 = a169 * a400;
    TScalar a412 = -a228 - a367 * a6;
    TScalar a413 = a175 * a400;
    TScalar a414 = a182 * a400;
    TScalar a415 = a12 * a398;
    TScalar a416 = a398 * a4;
    TScalar a417 = -a383 - a416;
    TScalar a418 = a131 * a21;
    TScalar a419 = a187 * a400;
    TScalar a420 = a193 * a400;
    TScalar a421 = a13 * a367;
    TScalar a422 = a13 * a131;
    TScalar a423 = a264 * a396 + a264 * a422 + a268 * a418 + a390 * a85;
    TScalar a424 = -a191 - a38 * a398;
    TScalar a425 = a144 * a85;
    TScalar a426 = a143 * a425;
    TScalar a427 = (TScalar(1) / TScalar(2)) * a84;
    TScalar a428 = -a228 - a38 * a427;
    TScalar a429 = a152 * a425;
    TScalar a430 = -a402 - a427 * a47;
    TScalar a431 = a158 * a425;
    TScalar a432 = -a23 - a29 * a398;
    TScalar a433 = a162 * a425;
    TScalar a434 = -a29 * a427 - a33;
    TScalar a435 = a169 * a425;
    TScalar a436 = -a409 - a427 * a6;
    TScalar a437 = a175 * a425;
    TScalar a438 = a182 * a425;
    TScalar a439 = a21 * a398;
    TScalar a440 = a139 * a21;
    TScalar a441 = a187 * a425;
    TScalar a442 = a21 * a427;
    TScalar a443 = -a13 * a427 - a416;
    TScalar a444 = 2 * a268;
    TScalar a445 = a193 * a425;
    math::linalg::mini::SMatrix<TScalar, kDofs, kDofs> hess_d;
    hess_d[0]  = 0;
    hess_d[1]  = 0;
    hess_d[2]  = 0;
    hess_d[3]  = a37;
    hess_d[4]  = a45;
    hess_d[5]  = a52;
    hess_d[6]  = a60;
    hess_d[7]  = a67;
    hess_d[8]  = a71;
    hess_d[9]  = a76;
    hess_d[10] = a82;
    hess_d[11] = a87;
    hess_d[12] = 0;
    hess_d[13] = 0;
    hess_d[14] = 0;
    hess_d[15] = a91;
    hess_d[16] = a92;
    hess_d[17] = a94;
    hess_d[18] = a97;
    hess_d[19] = a98;
    hess_d[20] = a101;
    hess_d[21] = a104;
    hess_d[22] = a105;
    hess_d[23] = a108;
    hess_d[24] = 0;
    hess_d[25] = 0;
    hess_d[26] = 0;
    hess_d[27] = a110;
    hess_d[28] = a112;
    hess_d[29] = a113;
    hess_d[30] = a116;
    hess_d[31] = a119;
    hess_d[32] = a120;
    hess_d[33] = a123;
    hess_d[34] = a126;
    hess_d[35] = a127;
    hess_d[36] = a37;
    hess_d[37] = a91;
    hess_d[38] = a110;
    hess_d[39] = a130 * a132 + a130 * a134 + a130 * a136 + 2 * a137 * a138 + 2 * a140 * a28 +
                 a146 * a147 + a146 * a148 + a146 * a149 - 2 * a37;
    hess_d[40] = a131 * a144 * a152 * a28 * a33 - a132 * a150 + a133 * a144 * a152 * a28 * a89 -
                 a134 * a150 + a135 * a144 * a15 * a152 * a28 - a151 * a17 - a153;
    hess_d[41] = a131 * a144 * a158 * a28 * a33 - a132 * a154 + a133 * a144 * a158 * a28 * a89 -
                 a134 * a154 + a135 * a144 * a15 * a158 * a28 - a151 * a155 - a159;
    hess_d[42] =
        a132 * a161 + a134 * a161 + a136 * a161 + a147 * a163 + a148 * a163 + a149 * a163 + a167;
    hess_d[43] =
        a132 * a168 + a134 * a168 + a136 * a168 + a147 * a170 + a148 * a170 + a149 * a170 + a173;
    hess_d[44] =
        a132 * a174 + a134 * a174 + a136 * a174 + a147 * a176 + a148 * a176 + a149 * a176 + a179;
    hess_d[45] =
        a132 * a181 + a134 * a181 + a136 * a181 + a147 * a183 + a148 * a183 + a149 * a183 + a184;
    hess_d[46] =
        a132 * a186 + a134 * a186 + a136 * a186 + a147 * a188 + a148 * a188 + a149 * a188 + a190;
    hess_d[47] =
        a132 * a192 + a134 * a192 + a136 * a192 + a147 * a194 + a148 * a194 + a149 * a194 + a195;
    hess_d[48] = a45;
    hess_d[49] = a92;
    hess_d[50] = a112;
    hess_d[51] = a131 * a143 * a144 * a33 * a44 - a132 * a197 + a133 * a143 * a144 * a44 * a89 -
                 a134 * a197 + a135 * a143 * a144 * a15 * a44 - a140 * a43 - a153;
    hess_d[52] = a132 * a199 + a134 * a199 + a136 * a199 + a147 * a203 + a148 * a203 + a149 * a203 +
                 2 * a151 * a44 + 2 * a200 * a201 - 2 * a92;
    hess_d[53] = a131 * a144 * a158 * a33 * a44 - a132 * a204 + a133 * a144 * a158 * a44 * a89 -
                 a134 * a204 + a135 * a144 * a15 * a158 * a44 - a139 * a155 * a47 - a205;
    hess_d[54] =
        a132 * a206 + a134 * a206 + a136 * a206 + a147 * a207 + a148 * a207 + a149 * a207 + a211;
    hess_d[55] =
        a132 * a213 + a134 * a213 + a136 * a213 + a147 * a214 + a148 * a214 + a149 * a214 + a216;
    hess_d[56] =
        a132 * a217 + a134 * a217 + a136 * a217 + a147 * a218 + a148 * a218 + a149 * a218 + a220;
    hess_d[57] =
        a132 * a221 + a134 * a221 + a136 * a221 + a147 * a222 + a148 * a222 + a149 * a222 + a223;
    hess_d[58] =
        a132 * a225 + a134 * a225 + a136 * a225 + a147 * a226 + a148 * a226 + a149 * a226 + a227;
    hess_d[59] =
        a132 * a229 + a134 * a229 + a136 * a229 + a147 * a230 + a148 * a230 + a149 * a230 + a232;
    hess_d[60] = a52;
    hess_d[61] = a94;
    hess_d[62] = a113;
    hess_d[63] = a131 * a143 * a144 * a33 * a51 - a132 * a233 + a133 * a143 * a144 * a51 * a89 -
                 a134 * a233 + a135 * a143 * a144 * a15 * a51 - a159 - a234 * a43;
    hess_d[64] = a131 * a144 * a152 * a33 * a51 - a132 * a236 + a133 * a144 * a152 * a51 * a89 -
                 a134 * a236 + a135 * a144 * a15 * a152 * a51 - a16 * a234 * a49 - a205;
    hess_d[65] = -2 * a113 + a132 * a237 + a134 * a237 + a136 * a237 + a147 * a242 + a148 * a242 +
                 a149 * a242 + 2 * a238 * a239 + 2 * a240 * a93;
    hess_d[66] =
        a132 * a243 + a134 * a243 + a136 * a243 + a147 * a244 + a148 * a244 + a149 * a244 + a248;
    hess_d[67] =
        a132 * a249 + a134 * a249 + a136 * a249 + a147 * a250 + a148 * a250 + a149 * a250 + a253;
    hess_d[68] =
        a132 * a254 + a134 * a254 + a136 * a254 + a147 * a255 + a148 * a255 + a149 * a255 + a257;
    hess_d[69] =
        a132 * a258 + a134 * a258 + a136 * a258 + a147 * a259 + a148 * a259 + a149 * a259 + a261;
    hess_d[70] =
        a132 * a262 + a134 * a262 + a136 * a262 + a147 * a263 + a148 * a263 + a149 * a263 + a265;
    hess_d[71] =
        a132 * a266 + a134 * a266 + a136 * a266 + a147 * a267 + a148 * a267 + a149 * a267 + a269;
    hess_d[72] = a60;
    hess_d[73] = a97;
    hess_d[74] = a116;
    hess_d[75] =
        a132 * a273 + a134 * a273 + a136 * a273 + a147 * a275 + a148 * a275 + a149 * a275 + a167;
    hess_d[76] =
        a132 * a276 + a134 * a276 + a136 * a276 + a147 * a277 + a148 * a277 + a149 * a277 + a211;
    hess_d[77] =
        a132 * a278 + a134 * a278 + a136 * a278 + a147 * a279 + a148 * a279 + a149 * a279 + a248;
    hess_d[78] = a132 * a281 + a134 * a281 + a136 * a281 + a147 * a284 + a148 * a284 + a149 * a284 +
                 2 * a245 * a283 + 2 * a282 * a7;
    hess_d[79] =
        -a132 * a286 - a134 * a286 + a147 * a285 + a148 * a285 + a149 * a285 - a287 * a56 + a290;
    hess_d[80] =
        -a132 * a292 - a134 * a292 + a147 * a291 + a148 * a291 + a149 * a291 - a287 * a293 + a295;
    hess_d[81] =
        a132 * a297 + a134 * a297 + a136 * a297 + a147 * a298 + a148 * a298 + a149 * a298 + a299;
    hess_d[82] =
        a132 * a300 + a134 * a300 + a136 * a300 + a147 * a301 + a148 * a301 + a149 * a301 + a302;
    hess_d[83] =
        a132 * a303 + a134 * a303 + a136 * a303 + a147 * a304 + a148 * a304 + a149 * a304 + a305;
    hess_d[84] = a67;
    hess_d[85] = a98;
    hess_d[86] = a119;
    hess_d[87] =
        a132 * a307 + a134 * a307 + a136 * a307 + a147 * a309 + a148 * a309 + a149 * a309 + a173;
    hess_d[88] =
        a132 * a311 + a134 * a311 + a136 * a311 + a147 * a312 + a148 * a312 + a149 * a312 + a216;
    hess_d[89] =
        a132 * a313 + a134 * a313 + a136 * a313 + a147 * a314 + a148 * a314 + a149 * a314 + a253;
    hess_d[90] =
        -a132 * a316 - a134 * a316 + a147 * a315 + a148 * a315 + a149 * a315 - a288 * a64 + a290;
    hess_d[91] = a132 * a318 + a134 * a318 + a136 * a318 + a147 * a320 + a148 * a320 + a149 * a320 +
                 2 * a287 * a65 + 2 * a29 * a319;
    hess_d[92] = -a132 * a322 - a134 * a322 - a139 * a293 * a6 + a147 * a321 + a148 * a321 +
                 a149 * a321 + a325;
    hess_d[93] =
        a132 * a326 + a134 * a326 + a136 * a326 + a147 * a327 + a148 * a327 + a149 * a327 + a329;
    hess_d[94] =
        a132 * a331 + a134 * a331 + a136 * a331 + a147 * a332 + a148 * a332 + a149 * a332 + a333;
    hess_d[95] =
        a132 * a334 + a134 * a334 + a136 * a334 + a147 * a335 + a148 * a335 + a149 * a335 + a336;
    hess_d[96] = a71;
    hess_d[97] = a101;
    hess_d[98] = a120;
    hess_d[99] =
        a132 * a337 + a134 * a337 + a136 * a337 + a147 * a339 + a148 * a339 + a149 * a339 + a179;
    hess_d[100] =
        a132 * a341 + a134 * a341 + a136 * a341 + a147 * a342 + a148 * a342 + a149 * a342 + a220;
    hess_d[101] =
        a132 * a343 + a134 * a343 + a136 * a343 + a147 * a344 + a148 * a344 + a149 * a344 + a257;
    hess_d[102] =
        -a132 * a346 - a134 * a346 + a147 * a345 + a148 * a345 + a149 * a345 + a295 - a347 * a64;
    hess_d[103] = -a132 * a349 - a134 * a349 + a147 * a348 + a148 * a348 + a149 * a348 -
                  a16 * a347 * a54 + a325;
    hess_d[104] = 2 * a11 * a351 + a132 * a350 + a134 * a350 + a136 * a350 + a147 * a353 +
                  a148 * a353 + a149 * a353 + 2 * a256 * a352;
    hess_d[105] =
        a132 * a354 + a134 * a354 + a136 * a354 + a147 * a355 + a148 * a355 + a149 * a355 + a358;
    hess_d[106] =
        a132 * a359 + a134 * a359 + a136 * a359 + a147 * a360 + a148 * a360 + a149 * a360 + a362;
    hess_d[107] =
        a132 * a363 + a134 * a363 + a136 * a363 + a147 * a364 + a148 * a364 + a149 * a364 + a365;
    hess_d[108] = a76;
    hess_d[109] = a104;
    hess_d[110] = a123;
    hess_d[111] =
        a132 * a369 + a134 * a369 + a136 * a369 + a147 * a371 + a148 * a371 + a149 * a371 + a184;
    hess_d[112] =
        a132 * a372 + a134 * a372 + a136 * a372 + a147 * a373 + a148 * a373 + a149 * a373 + a223;
    hess_d[113] =
        a132 * a374 + a134 * a374 + a136 * a374 + a147 * a375 + a148 * a375 + a149 * a375 + a261;
    hess_d[114] =
        a132 * a377 + a134 * a377 + a136 * a377 + a147 * a378 + a148 * a378 + a149 * a378 + a299;
    hess_d[115] =
        a132 * a379 + a134 * a379 + a136 * a379 + a147 * a380 + a148 * a380 + a149 * a380 + a329;
    hess_d[116] =
        a132 * a381 + a134 * a381 + a136 * a381 + a147 * a382 + a148 * a382 + a149 * a382 + a358;
    hess_d[117] = a132 * a384 + a134 * a384 + a136 * a384 + a147 * a387 + a148 * a387 +
                  a149 * a387 + 2 * a260 * a385 + 2 * a386 * a75;
    hess_d[118] =
        -a132 * a389 - a134 * a389 + a147 * a388 + a148 * a388 + a149 * a388 - a390 * a73 + a392;
    hess_d[119] =
        -a132 * a394 - a134 * a394 + a147 * a393 + a148 * a393 + a149 * a393 - a390 * a395 + a397;
    hess_d[120] = a82;
    hess_d[121] = a105;
    hess_d[122] = a126;
    hess_d[123] =
        a132 * a399 + a134 * a399 + a136 * a399 + a147 * a401 + a148 * a401 + a149 * a401 + a190;
    hess_d[124] =
        a132 * a403 + a134 * a403 + a136 * a403 + a147 * a404 + a148 * a404 + a149 * a404 + a227;
    hess_d[125] =
        a132 * a405 + a134 * a405 + a136 * a405 + a147 * a406 + a148 * a406 + a149 * a406 + a265;
    hess_d[126] =
        a132 * a407 + a134 * a407 + a136 * a407 + a147 * a408 + a148 * a408 + a149 * a408 + a302;
    hess_d[127] =
        a132 * a410 + a134 * a410 + a136 * a410 + a147 * a411 + a148 * a411 + a149 * a411 + a333;
    hess_d[128] =
        a132 * a412 + a134 * a412 + a136 * a412 + a147 * a413 + a148 * a413 + a149 * a413 + a362;
    hess_d[129] =
        -a132 * a415 - a134 * a415 + a147 * a414 + a148 * a414 + a149 * a414 - a386 * a79 + a392;
    hess_d[130] = a132 * a417 + a134 * a417 + a136 * a417 + a147 * a419 + a148 * a419 +
                  a149 * a419 + 2 * a264 * a418 + 2 * a390 * a80;
    hess_d[131] = -a13 * a139 * a395 - a132 * a421 - a134 * a421 + a147 * a420 + a148 * a420 +
                  a149 * a420 + a423;
    hess_d[132] = a87;
    hess_d[133] = a108;
    hess_d[134] = a127;
    hess_d[135] =
        a132 * a424 + a134 * a424 + a136 * a424 + a147 * a426 + a148 * a426 + a149 * a426 + a195;
    hess_d[136] =
        a132 * a428 + a134 * a428 + a136 * a428 + a147 * a429 + a148 * a429 + a149 * a429 + a232;
    hess_d[137] =
        a132 * a430 + a134 * a430 + a136 * a430 + a147 * a431 + a148 * a431 + a149 * a431 + a269;
    hess_d[138] =
        a132 * a432 + a134 * a432 + a136 * a432 + a147 * a433 + a148 * a433 + a149 * a433 + a305;
    hess_d[139] =
        a132 * a434 + a134 * a434 + a136 * a434 + a147 * a435 + a148 * a435 + a149 * a435 + a336;
    hess_d[140] =
        a132 * a436 + a134 * a436 + a136 * a436 + a147 * a437 + a148 * a437 + a149 * a437 + a365;
    hess_d[141] =
        -a132 * a439 - a134 * a439 + a147 * a438 + a148 * a438 + a149 * a438 + a397 - a440 * a79;
    hess_d[142] = -a132 * a442 - a134 * a442 + a147 * a441 + a148 * a441 + a149 * a441 -
                  a16 * a440 * a84 + a423;
    hess_d[143] = a132 * a443 + a134 * a443 + a136 * a443 + a147 * a445 + a148 * a445 +
                  a149 * a445 + a396 * a444 + a422 * a444;
    return hess_d;
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline TScalar EdgeEdgeDistance<TScalar>::Eval(TMatrixx const& x_, ScalarType eps)
{
    auto a = x_.template Slice<3, 1>(0, 0);
    auto b = x_.template Slice<3, 1>(3, 0);
    auto c = x_.template Slice<3, 1>(6, 0);
    auto d = x_.template Slice<3, 1>(9, 0);
    using namespace std;
    // NOTE: We use the geometry::DistanceQueries::LineSegments function, because
    // it handles degenerate and parallel edges.
    return sqrt(geometry::DistanceQueries::LineSegments(a, b, c, d, eps));
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline auto EdgeEdgeDistance<TScalar>::Gradient(TMatrixx const& x_, ScalarType eps)
    -> math::linalg::mini::SVector<TScalar, kDofs>
{
    auto a = x_.template Slice<3, 1>(0, 0);
    auto b = x_.template Slice<3, 1>(3, 0);
    auto c = x_.template Slice<3, 1>(6, 0);
    auto d = x_.template Slice<3, 1>(9, 0);
    math::linalg::mini::SVector<TScalar, kDofs> g;
    auto ga = g.template Slice<3, 1>(0, 0);
    auto gb = g.template Slice<3, 1>(3, 0);
    auto gc = g.template Slice<3, 1>(6, 0);
    auto gd = g.template Slice<3, 1>(9, 0);
    using namespace std;
    // NOTE: This is the actual edge-edge distance function, but it's too unstable
    // math::linalg::mini::SVector<TScalar, 3> ab = b - a;
    // math::linalg::mini::SVector<TScalar, 3> cd = d - c;
    // math::linalg::mini::SVector<TScalar, 3> ac = c - a;
    // math::linalg::mini::SVector<TScalar, 3> n  = Cross(ab, cd);
    // TScalar nnorm                              = sqrt(SquaredNorm(n) + eps * eps);
    // TScalar nnorminv                           = 1 / nnorm;
    // n *= nnorminv;
    // math::linalg::mini::SVector<TScalar, 3> Pnac = ac - Dot(ac, n) * n;
    // ga      = nnorminv * Cross(-cd, Pnac);
    // gb      = -ga;
    // ga -= n;
    // gc = nnorminv * Cross(ab, Pnac);
    // gd = -gc;
    // gc += n;

    // NOTE: This is a fake edge-edge distance function that only takes into account linear
    // displacements
    math::linalg::mini::SVector<TScalar, 2> st = ClosestPointQueries::LineSegments(a, b, c, d);
    math::linalg::mini::SVector<TScalar, 3> n =
        ((1 - st(1)) * c + st(1) * d) - ((1 - st(0)) * a + st(0) * b);
    TScalar nnorm    = Norm(n);
    TScalar invnnorm = 1 / nnorm;
    n *= invnnorm;
    ga = -(1 - st(0)) * n;
    gb = -st(0) * n;
    gc = (1 - st(1)) * n;
    gd = st(1) * n;
    return g;
}

template <common::CFloatingPoint TScalar>
template <math::linalg::mini::CMatrix TMatrixx>
inline auto EdgeEdgeDistance<TScalar>::Hessian(TMatrixx const& x_, ScalarType eps)
    -> math::linalg::mini::SMatrix<TScalar, kDofs, kDofs>
{
    auto a = x_.template Slice<3, 1>(0, 0);
    auto b = x_.template Slice<3, 1>(3, 0);
    auto c = x_.template Slice<3, 1>(6, 0);
    auto d = x_.template Slice<3, 1>(9, 0);
    using namespace std;
    TScalar a0  = c[1] - d[1];
    TScalar a1  = 2 * c[1] - 2 * d[1];
    TScalar a2  = (TScalar(1) / TScalar(2)) * a1;
    TScalar a3  = a0 * a2;
    TScalar a4  = c[2] - d[2];
    TScalar a5  = -a4;
    TScalar a6  = 2 * c[2] - 2 * d[2];
    TScalar a7  = -a6;
    TScalar a8  = (TScalar(1) / TScalar(2)) * a7;
    TScalar a9  = a5 * a8;
    TScalar a10 = -a3 - a9;
    TScalar a11 = -a[0] + c[0];
    TScalar a12 = a[1] - b[1];
    TScalar a13 = -a12;
    TScalar a14 = a13 * a5;
    TScalar a15 = a[2] - b[2];
    TScalar a16 = -a15;
    TScalar a17 = -a0;
    TScalar a18 = a16 * a17;
    TScalar a19 = a14 - a18;
    TScalar a20 = a[0] - b[0];
    TScalar a21 = -a20;
    TScalar a22 = a17 * a21;
    TScalar a23 = c[0] - d[0];
    TScalar a24 = -a23;
    TScalar a25 = a13 * a24;
    TScalar a26 = a22 - a25;
    TScalar a27 = a21 * a5;
    TScalar a28 = a16 * a24;
    TScalar a29 = a27 - a28;
    TScalar a30 = -a29;
    TScalar a31 = ((a19) * (a19)) + ((a26) * (a26)) + ((a30) * (a30)) + ((eps) * (eps));
    TScalar a32 = pow(a31, -TScalar(3) / TScalar(2));
    TScalar a33 = a19 * a32;
    TScalar a34 = a11 * a33;
    TScalar a35 = -a[1] + c[1];
    TScalar a36 = a30 * a32;
    TScalar a37 = a35 * a36;
    TScalar a38 = -a[2] + c[2];
    TScalar a39 = a26 * a32;
    TScalar a40 = a38 * a39;
    TScalar a41 = a35 * a5;
    TScalar a42 = a2 * a26;
    TScalar a43 = a30 * a8;
    TScalar a44 = -a42 - a43;
    TScalar a45 = a32 * a44;
    TScalar a46 = 2 * a45;
    TScalar a47 = a0 * a38;
    TScalar a48 = a33 * a44;
    TScalar a49 = (TScalar(3) / TScalar(2)) * a26;
    TScalar a50 = (TScalar(3) / TScalar(2)) * a30;
    TScalar a51 = -a1 * a49 - a50 * a7;
    TScalar a52 = pow(a31, -TScalar(5) / TScalar(2));
    TScalar a53 = a44 * a52;
    TScalar a54 = a51 * a53;
    TScalar a55 = a11 * a19;
    TScalar a56 = a30 * a35;
    TScalar a57 = a26 * a38;
    TScalar a58 = a2 * a24;
    TScalar a59 = a24 * a38;
    TScalar a60 = a32 * a42;
    TScalar a61 = 2 * c[0] - 2 * d[0];
    TScalar a62 = -a61;
    TScalar a63 = a19 * a6;
    TScalar a64 = -a49 * a62 - TScalar(3) / TScalar(2) * a63;
    TScalar a65 = (TScalar(1) / TScalar(2)) * a26;
    TScalar a66 = a62 * a65;
    TScalar a67 = (TScalar(1) / TScalar(2)) * a63;
    TScalar a68 = -a66 - a67;
    TScalar a69 = 1 / sqrt(a31);
    TScalar a70 = a5 * a69;
    TScalar a71 = a30 * a45 + a70;
    TScalar a72 = a4 * a69;
    TScalar a73 = a33 * a68 + a72;
    TScalar a74 = -a0 * a32 * a38 * a68 - a11 * a32 * a4 * a44 - a24 * a32 * a38 * a44 -
                  a32 * a35 * a5 * a68 + a71 + a73;
    TScalar a75 = a23 * a8;
    TScalar a76 = a23 * a35;
    TScalar a77 = a32 * a43;
    TScalar a78 = -a1;
    TScalar a79 = a19 * a78;
    TScalar a80 = -a50 * a61 - TScalar(3) / TScalar(2) * a79;
    TScalar a81 = (TScalar(1) / TScalar(2)) * a30;
    TScalar a82 = a61 * a81;
    TScalar a83 = -TScalar(1) / TScalar(2) * a79 - a82;
    TScalar a84 = a0 * a69;
    TScalar a85 = a26 * a45 + a84;
    TScalar a86 = a17 * a69;
    TScalar a87 = a33 * a83 + a86;
    TScalar a88 = -a0 * a32 * a38 * a83 - a11 * a17 * a32 * a44 - a23 * a32 * a35 * a44 -
                  a32 * a35 * a5 * a83 + a85 + a87;
    TScalar a89  = a17 * a2;
    TScalar a90  = a4 * a8;
    TScalar a91  = -a89 - a90;
    TScalar a92  = -a49 * a78 - a50 * a6;
    TScalar a93  = a53 * a92;
    TScalar a94  = a65 * a78;
    TScalar a95  = a6 * a81;
    TScalar a96  = -a94 - a95;
    TScalar a97  = a33 * a96;
    TScalar a98  = a35 * a45;
    TScalar a99  = a32 * a96;
    TScalar a100 = a38 * a45;
    TScalar a101 = a100 * a17 + a4 * a98 + a41 * a99 + a47 * a99 - a97;
    TScalar a102 = (TScalar(3) / TScalar(2)) * a19;
    TScalar a103 = -a102 * a7 - a49 * a61;
    TScalar a104 = a103 * a53;
    TScalar a105 = a2 * a23;
    TScalar a106 = a23 * a38;
    TScalar a107 = a61 * a65;
    TScalar a108 = -a107 - a19 * a8;
    TScalar a109 = a108 * a33;
    TScalar a110 = a11 * a45;
    TScalar a111 = a108 * a32;
    TScalar a112 = a100 * a23 - a109 + a110 * a5 + a111 * a41 + a111 * a47 - a70;
    TScalar a113 = -a1 * a102 - a50 * a62;
    TScalar a114 = a113 * a53;
    TScalar a115 = a24 * a8;
    TScalar a116 = a24 * a35;
    TScalar a117 = a62 * a81;
    TScalar a118 = -a117 - a19 * a2;
    TScalar a119 = a118 * a33;
    TScalar a120 = a118 * a32;
    TScalar a121 = a0 * a110 - a119 + a120 * a41 + a120 * a47 + a24 * a98 - a84;
    TScalar a122 = a13 * a2;
    TScalar a123 = a15 * a8;
    TScalar a124 = -a122 - a123;
    TScalar a125 = 2 * a[1] - 2 * b[1];
    TScalar a126 = -a125;
    TScalar a127 = 2 * a[2] - 2 * b[2];
    TScalar a128 = -a126 * a49 - a127 * a50;
    TScalar a129 = a128 * a53;
    TScalar a130 = a126 * a65;
    TScalar a131 = a127 * a81;
    TScalar a132 = -a130 - a131;
    TScalar a133 = a132 * a33;
    TScalar a134 = a132 * a32;
    TScalar a135 = a100 * a13 - a133 + a134 * a41 + a134 * a47 + a15 * a98 + a48;
    TScalar a136 = -a2 * a20 - a26;
    TScalar a137 = 2 * a[0] - 2 * b[0];
    TScalar a138 = -a127;
    TScalar a139 = -a102 * a138 - a137 * a49;
    TScalar a140 = a139 * a53;
    TScalar a141 = a137 * a65;
    TScalar a142 = (TScalar(1) / TScalar(2)) * a19;
    TScalar a143 = -a138 * a142 - a141;
    TScalar a144 = a143 * a33;
    TScalar a145 = a143 * a32;
    TScalar a146 = a38 * a69;
    TScalar a147 = a16 * a69;
    TScalar a148 = a146 - a147;
    TScalar a149 = a100 * a20 + a110 * a16 - a144 + a145 * a41 + a145 * a47 + a148 + a71;
    TScalar a150 = -a21 * a8 - a29;
    TScalar a151 = -a137;
    TScalar a152 = -a102 * a125 - a151 * a50;
    TScalar a153 = a152 * a53;
    TScalar a154 = a151 * a81;
    TScalar a155 = -a125 * a142 - a154;
    TScalar a156 = a155 * a33;
    TScalar a157 = a155 * a32;
    TScalar a158 = a35 * a69;
    TScalar a159 = -a158;
    TScalar a160 = a12 * a69;
    TScalar a161 = a159 - a160;
    TScalar a162 = a110 * a12 - a156 + a157 * a41 + a157 * a47 + a161 + a21 * a98 + a85;
    TScalar a163 = a12 * a2;
    TScalar a164 = a16 * a8;
    TScalar a165 = -a163 - a164;
    TScalar a166 = -a125 * a49 - a138 * a50;
    TScalar a167 = a166 * a53;
    TScalar a168 = a125 * a65;
    TScalar a169 = a138 * a81;
    TScalar a170 = -a168 - a169;
    TScalar a171 = a170 * a33;
    TScalar a172 = a170 * a32;
    TScalar a173 = a100 * a12 + a16 * a98 - a171 + a172 * a41 + a172 * a47;
    TScalar a174 = -a22 + a25;
    TScalar a175 = -a174 - a2 * a21;
    TScalar a176 = -a102 * a127 - a151 * a49;
    TScalar a177 = a176 * a53;
    TScalar a178 = a151 * a65;
    TScalar a179 = -a127 * a142 - a178;
    TScalar a180 = a179 * a33;
    TScalar a181 = a179 * a32;
    TScalar a182 = -a146;
    TScalar a183 = a15 * a69;
    TScalar a184 = a182 - a183;
    TScalar a185 = a100 * a21 + a110 * a15 - a180 + a181 * a41 + a181 * a47 + a184;
    TScalar a186 = -a27 + a28;
    TScalar a187 = -a186 - a20 * a8;
    TScalar a188 = -a102 * a126 - a137 * a50;
    TScalar a189 = a188 * a53;
    TScalar a190 = a137 * a81;
    TScalar a191 = -a126 * a142 - a190;
    TScalar a192 = a191 * a33;
    TScalar a193 = a191 * a32;
    TScalar a194 = a13 * a69;
    TScalar a195 = a158 - a194;
    TScalar a196 = a110 * a13 - a192 + a193 * a41 + a193 * a47 + a195 + a20 * a98;
    TScalar a197 = (TScalar(1) / TScalar(2)) * a62;
    TScalar a198 = a197 * a34;
    TScalar a199 = a32 * a35;
    TScalar a200 = a117 * a199;
    TScalar a201 = a32 * a66;
    TScalar a202 = a197 * a24;
    TScalar a203 = (TScalar(1) / TScalar(2)) * a6;
    TScalar a204 = a203 * a4;
    TScalar a205 = -a202 - a204;
    TScalar a206 = a11 * a4;
    TScalar a207 = a32 * a68;
    TScalar a208 = 2 * a207;
    TScalar a209 = a207 * a30;
    TScalar a210 = a52 * a68;
    TScalar a211 = a210 * a64;
    TScalar a212 = a11 * a17;
    TScalar a213 = a32 * a67;
    TScalar a214 = a199 * a95;
    TScalar a215 = a17 * a38;
    TScalar a216 = a215 * a32;
    TScalar a217 = a6 * a65;
    TScalar a218 = a24 * a69;
    TScalar a219 = a207 * a26 + a218;
    TScalar a220 = a23 * a69;
    TScalar a221 = a220 + a36 * a83;
    TScalar a222 = -a11 * a17 * a32 * a68 - a11 * a32 * a4 * a83 + a219 + a221 -
                   a23 * a32 * a35 * a68 - a24 * a32 * a38 * a83;
    TScalar a223 = a210 * a92;
    TScalar a224 = a36 * a96;
    TScalar a225 = a207 * a35;
    TScalar a226 = a17 * a207 * a38 + a206 * a99 - a224 + a225 * a4 + a59 * a99 - a72;
    TScalar a227 = a197 * a23;
    TScalar a228 = a203 * a5;
    TScalar a229 = -a227 - a228;
    TScalar a230 = a103 * a210;
    TScalar a231 = a108 * a36;
    TScalar a232 = a11 * a207;
    TScalar a233 = a207 * a38;
    TScalar a234 = a111 * a206 + a111 * a59 + a23 * a233 - a231 + a232 * a5;
    TScalar a235 = a113 * a210;
    TScalar a236 = a0 * a11;
    TScalar a237 = a217 * a32;
    TScalar a238 = a118 * a36;
    TScalar a239 = a0 * a232 + a120 * a206 + a120 * a59 - a218 + a225 * a24 - a238;
    TScalar a240 = -a13 * a197 - a174;
    TScalar a241 = a128 * a210;
    TScalar a242 = a132 * a36;
    TScalar a243 = a13 * a233 + a134 * a206 + a134 * a59 + a15 * a225 + a184 - a242 + a73;
    TScalar a244 = a197 * a20;
    TScalar a245 = a16 * a203;
    TScalar a246 = -a244 - a245;
    TScalar a247 = a139 * a210;
    TScalar a248 = a143 * a36;
    TScalar a249 = a145 * a206 + a145 * a59 + a16 * a232 + a20 * a233 + a209 - a248;
    TScalar a250 = -a12 * a203 - a19;
    TScalar a251 = a152 * a210;
    TScalar a252 = a155 * a36;
    TScalar a253 = a11 * a69;
    TScalar a254 = a21 * a69;
    TScalar a255 = a253 - a254;
    TScalar a256 = a12 * a232 + a157 * a206 + a157 * a59 + a21 * a225 + a219 - a252 + a255;
    TScalar a257 = -a12 * a197 - a26;
    TScalar a258 = a166 * a210;
    TScalar a259 = a170 * a36;
    TScalar a260 = a12 * a233 + a148 + a16 * a225 + a172 * a206 + a172 * a59 - a259;
    TScalar a261 = a197 * a21;
    TScalar a262 = a15 * a203;
    TScalar a263 = -a261 - a262;
    TScalar a264 = a176 * a210;
    TScalar a265 = a179 * a36;
    TScalar a266 = a15 * a232 + a181 * a206 + a181 * a59 + a21 * a233 - a265;
    TScalar a267 = -a14 + a18;
    TScalar a268 = -a13 * a203 - a267;
    TScalar a269 = a188 * a210;
    TScalar a270 = a191 * a36;
    TScalar a271 = -a253;
    TScalar a272 = a20 * a69;
    TScalar a273 = a271 - a272;
    TScalar a274 = a13 * a232 + a193 * a206 + a193 * a59 + a20 * a225 - a270 + a273;
    TScalar a275 = (TScalar(1) / TScalar(2)) * a61;
    TScalar a276 = a275 * a34;
    TScalar a277 = a32 * a82;
    TScalar a278 = a32 * a38;
    TScalar a279 = a107 * a278;
    TScalar a280 = (TScalar(1) / TScalar(2)) * a78;
    TScalar a281 = a280 * a34;
    TScalar a282 = a35 * a4;
    TScalar a283 = a282 * a32;
    TScalar a284 = a78 * a81;
    TScalar a285 = a278 * a94;
    TScalar a286 = a23 * a275;
    TScalar a287 = a17 * a280;
    TScalar a288 = -a286 - a287;
    TScalar a289 = a32 * a83;
    TScalar a290 = 2 * a289;
    TScalar a291 = a39 * a83;
    TScalar a292 = a52 * a83;
    TScalar a293 = a292 * a80;
    TScalar a294 = a292 * a92;
    TScalar a295 = a39 * a96;
    TScalar a296 = a289 * a35;
    TScalar a297 = a289 * a38;
    TScalar a298 = a17 * a297 + a212 * a99 - a295 + a296 * a4 + a76 * a99 - a86;
    TScalar a299 = a103 * a292;
    TScalar a300 = a284 * a32;
    TScalar a301 = a108 * a39;
    TScalar a302 = a11 * a289;
    TScalar a303 = a111 * a212 + a111 * a76 - a220 + a23 * a297 - a301 + a302 * a5;
    TScalar a304 = a24 * a275;
    TScalar a305 = a0 * a280;
    TScalar a306 = -a304 - a305;
    TScalar a307 = a113 * a292;
    TScalar a308 = a118 * a39;
    TScalar a309 = a0 * a302 + a120 * a212 + a120 * a76 + a24 * a296 - a308;
    TScalar a310 = -a15 * a275 - a186;
    TScalar a311 = a128 * a292;
    TScalar a312 = a132 * a39;
    TScalar a313 = a13 * a297 + a134 * a212 + a134 * a76 + a15 * a296 + a195 - a312 + a87;
    TScalar a314 = -a16 * a280 - a267;
    TScalar a315 = a139 * a292;
    TScalar a316 = a143 * a39;
    TScalar a317 = a145 * a212 + a145 * a76 + a16 * a302 + a20 * a297 + a221 + a273 - a316;
    TScalar a318 = a21 * a275;
    TScalar a319 = a12 * a280;
    TScalar a320 = -a318 - a319;
    TScalar a321 = a152 * a292;
    TScalar a322 = a155 * a39;
    TScalar a323 = a12 * a302 + a157 * a212 + a157 * a76 + a21 * a296 + a291 - a322;
    TScalar a324 = -a16 * a275 - a29;
    TScalar a325 = a166 * a292;
    TScalar a326 = a170 * a39;
    TScalar a327 = a12 * a297 + a16 * a296 + a161 + a172 * a212 + a172 * a76 - a326;
    TScalar a328 = -a15 * a280 - a19;
    TScalar a329 = a176 * a292;
    TScalar a330 = a179 * a39;
    TScalar a331 = a15 * a302 + a181 * a212 + a181 * a76 + a21 * a297 + a255 - a330;
    TScalar a332 = a20 * a275;
    TScalar a333 = a13 * a280;
    TScalar a334 = -a332 - a333;
    TScalar a335 = a188 * a292;
    TScalar a336 = a191 * a39;
    TScalar a337 = a13 * a302 + a193 * a212 + a193 * a76 + a20 * a296 - a336;
    TScalar a338 = -a228 - a305;
    TScalar a339 = a52 * a96;
    TScalar a340 = a339 * a51;
    TScalar a341 = a339 * a64;
    TScalar a342 = a32 * a94;
    TScalar a343 = a339 * a80;
    TScalar a344 = a11 * a213;
    TScalar a345 = a32 * a95;
    TScalar a346 = -a204 - a287;
    TScalar a347 = a38 * a99;
    TScalar a348 = a339 * a92;
    TScalar a349 = a103 * a339;
    TScalar a350 = a11 * a99;
    TScalar a351 = a111 * a215 + a111 * a282 + a23 * a347 + a350 * a5;
    TScalar a352 = a113 * a339;
    TScalar a353 = a35 * a99;
    TScalar a354 = a0 * a350 + a120 * a215 + a120 * a282 + a24 * a353;
    TScalar a355 = -a262 - a333;
    TScalar a356 = a128 * a339;
    TScalar a357 = a13 * a347 + a134 * a215 + a134 * a282 + a15 * a353 + a97;
    TScalar a358 = -a174 - a20 * a280;
    TScalar a359 = a139 * a339;
    TScalar a360 = a145 * a215 + a145 * a282 + a16 * a350 + a182 + a20 * a347 + a224 + a72;
    TScalar a361 = -a186 - a203 * a21;
    TScalar a362 = a152 * a339;
    TScalar a363 = a12 * a350 + a157 * a215 + a157 * a282 + a158 + a21 * a353 + a295 + a86;
    TScalar a364 = -a245 - a319;
    TScalar a365 = a166 * a339;
    TScalar a366 = a12 * a347 + a16 * a353 + a172 * a215 + a172 * a282;
    TScalar a367 = -a21 * a280 - a26;
    TScalar a368 = a176 * a339;
    TScalar a369 = a146 + a15 * a350 + a181 * a215 + a181 * a282 + a21 * a347;
    TScalar a370 = -a20 * a203 - a29;
    TScalar a371 = a188 * a339;
    TScalar a372 = a13 * a350 + a159 + a193 * a215 + a193 * a282 + a20 * a353;
    TScalar a373 = a108 * a52;
    TScalar a374 = a373 * a51;
    TScalar a375 = a199 * a82;
    TScalar a376 = -a304 - a90;
    TScalar a377 = a373 * a64;
    TScalar a378 = a373 * a80;
    TScalar a379 = a17 * a8;
    TScalar a380 = a35 * a77;
    TScalar a381 = a373 * a92;
    TScalar a382 = -a286 - a9;
    TScalar a383 = a11 * a111;
    TScalar a384 = a103 * a373;
    TScalar a385 = a113 * a373;
    TScalar a386 = a0 * a8;
    TScalar a387 = a11 * a5;
    TScalar a388 = a111 * a35;
    TScalar a389 = a0 * a383 + a106 * a120 + a120 * a387 + a24 * a388;
    TScalar a390 = -a13 * a275 - a26;
    TScalar a391 = a128 * a373;
    TScalar a392 = a111 * a38;
    TScalar a393 = a106 * a134 + a109 + a13 * a392 + a134 * a387 + a146 + a15 * a388 + a70;
    TScalar a394 = -a164 - a332;
    TScalar a395 = a139 * a373;
    TScalar a396 = a106 * a145 + a145 * a387 + a16 * a383 + a20 * a392 + a231;
    TScalar a397 = -a12 * a8 - a267;
    TScalar a398 = a152 * a373;
    TScalar a399 = a106 * a157 + a12 * a383 + a157 * a387 + a21 * a388 + a220 + a271 + a301;
    TScalar a400 = -a12 * a275 - a174;
    TScalar a401 = a166 * a373;
    TScalar a402 = a106 * a172 + a12 * a392 + a16 * a388 + a172 * a387 + a182;
    TScalar a403 = -a123 - a318;
    TScalar a404 = a176 * a373;
    TScalar a405 = a106 * a181 + a15 * a383 + a181 * a387 + a21 * a392;
    TScalar a406 = -a13 * a8 - a19;
    TScalar a407 = a188 * a373;
    TScalar a408 = a106 * a193 + a13 * a383 + a193 * a387 + a20 * a388 + a253;
    TScalar a409 = a118 * a52;
    TScalar a410 = a409 * a51;
    TScalar a411 = a201 * a38;
    TScalar a412 = a409 * a64;
    TScalar a413 = a2 * a4;
    TScalar a414 = a38 * a60;
    TScalar a415 = -a227 - a89;
    TScalar a416 = a409 * a80;
    TScalar a417 = a409 * a92;
    TScalar a418 = a103 * a409;
    TScalar a419 = a2 * a5;
    TScalar a420 = -a202 - a3;
    TScalar a421 = a120 * a35;
    TScalar a422 = a113 * a409;
    TScalar a423 = -a15 * a197 - a29;
    TScalar a424 = a128 * a409;
    TScalar a425 = a120 * a38;
    TScalar a426 = a116 * a134 + a119 + a13 * a425 + a134 * a236 + a15 * a421 + a159 + a84;
    TScalar a427 = -a16 * a2 - a19;
    TScalar a428 = a139 * a409;
    TScalar a429 = a11 * a120;
    TScalar a430 = a116 * a145 + a145 * a236 + a16 * a429 + a20 * a425 + a218 + a238 + a253;
    TScalar a431 = -a163 - a261;
    TScalar a432 = a152 * a409;
    TScalar a433 = a116 * a157 + a12 * a429 + a157 * a236 + a21 * a421 + a308;
    TScalar a434 = -a16 * a197 - a186;
    TScalar a435 = a166 * a409;
    TScalar a436 = a116 * a172 + a12 * a425 + a158 + a16 * a421 + a172 * a236;
    TScalar a437 = -a15 * a2 - a267;
    TScalar a438 = a176 * a409;
    TScalar a439 = a116 * a181 + a15 * a429 + a181 * a236 + a21 * a425 + a271;
    TScalar a440 = -a122 - a244;
    TScalar a441 = a188 * a409;
    TScalar a442 = a116 * a193 + a13 * a429 + a193 * a236 + a20 * a421;
    TScalar a443 = (TScalar(1) / TScalar(2)) * a126;
    TScalar a444 = a0 * a443;
    TScalar a445 = (TScalar(1) / TScalar(2)) * a127;
    TScalar a446 = a445 * a5;
    TScalar a447 = -a444 - a446;
    TScalar a448 = a132 * a52;
    TScalar a449 = a448 * a51;
    TScalar a450 = -a174 - a24 * a443;
    TScalar a451 = a448 * a64;
    TScalar a452 = -a186 - a23 * a445;
    TScalar a453 = a448 * a80;
    TScalar a454 = a17 * a443;
    TScalar a455 = a4 * a445;
    TScalar a456 = -a454 - a455;
    TScalar a457 = a448 * a92;
    TScalar a458 = -a23 * a443 - a26;
    TScalar a459 = a103 * a448;
    TScalar a460 = -a24 * a445 - a29;
    TScalar a461 = a113 * a448;
    TScalar a462 = a13 * a443;
    TScalar a463 = a15 * a445;
    TScalar a464 = -a462 - a463;
    TScalar a465 = a134 * a38;
    TScalar a466 = a15 * a35;
    TScalar a467 = a128 * a448;
    TScalar a468 = a139 * a448;
    TScalar a469 = a34 * a443;
    TScalar a470 = a20 * a35;
    TScalar a471 = a32 * a470;
    TScalar a472 = a126 * a81;
    TScalar a473 = a20 * a38;
    TScalar a474 = a130 * a32;
    TScalar a475 = a13 * a38;
    TScalar a476 = a11 * a134;
    TScalar a477 = a144 + a145 * a466 + a145 * a475 + a147 + a16 * a476 + a183 + a20 * a465 + a242;
    TScalar a478 = a152 * a448;
    TScalar a479 = a34 * a445;
    TScalar a480 = a21 * a35;
    TScalar a481 = a32 * a480;
    TScalar a482 = a21 * a38;
    TScalar a483 = a32 * a65;
    TScalar a484 = a127 * a483;
    TScalar a485 = a134 * a35;
    TScalar a486 = a12 * a476 + a156 + a157 * a466 + a157 * a475 + a160 + a194 + a21 * a485 + a312;
    TScalar a487 = a12 * a443;
    TScalar a488 = a16 * a445;
    TScalar a489 = -a487 - a488;
    TScalar a490 = a166 * a448;
    TScalar a491 = a12 * a465 + a16 * a485 + a171 + a172 * a466 + a172 * a475;
    TScalar a492 = a176 * a448;
    TScalar a493 = a15 * a476 + a180 + a181 * a466 + a181 * a475 + a183 + a21 * a465;
    TScalar a494 = a188 * a448;
    TScalar a495 = a13 * a476 + a192 + a193 * a466 + a193 * a475 + a194 + a20 * a485;
    TScalar a496 = (TScalar(1) / TScalar(2)) * a137;
    TScalar a497 = -a0 * a496 - a26;
    TScalar a498 = a143 * a52;
    TScalar a499 = a498 * a51;
    TScalar a500 = a24 * a496;
    TScalar a501 = (TScalar(1) / TScalar(2)) * a138;
    TScalar a502 = a4 * a501;
    TScalar a503 = -a500 - a502;
    TScalar a504 = a498 * a64;
    TScalar a505 = -a17 * a501 - a267;
    TScalar a506 = a498 * a80;
    TScalar a507 = -a17 * a496 - a174;
    TScalar a508 = a498 * a92;
    TScalar a509 = a23 * a496;
    TScalar a510 = a5 * a501;
    TScalar a511 = -a509 - a510;
    TScalar a512 = a103 * a498;
    TScalar a513 = -a0 * a501 - a19;
    TScalar a514 = a113 * a498;
    TScalar a515 = a128 * a498;
    TScalar a516 = a34 * a496;
    TScalar a517 = a190 * a199;
    TScalar a518 = a141 * a32;
    TScalar a519 = a20 * a496;
    TScalar a520 = a16 * a501;
    TScalar a521 = -a519 - a520;
    TScalar a522 = a11 * a16;
    TScalar a523 = 2 * a145;
    TScalar a524 = a139 * a498;
    TScalar a525 = a152 * a498;
    TScalar a526 = a34 * a501;
    TScalar a527 = a169 * a199;
    TScalar a528 = a12 * a38;
    TScalar a529 = a138 * a483;
    TScalar a530 = a12 * a145;
    TScalar a531 = a145 * a35;
    TScalar a532 = a11 * a530 + a157 * a473 + a157 * a522 + a21 * a531 + a252 + a254 + a272 + a316;
    TScalar a533 = a166 * a498;
    TScalar a534 = a147 + a16 * a531 + a172 * a473 + a172 * a522 + a259 + a38 * a530;
    TScalar a535 = a21 * a496;
    TScalar a536 = a15 * a501;
    TScalar a537 = -a535 - a536;
    TScalar a538 = a176 * a498;
    TScalar a539 = a11 * a145;
    TScalar a540 = a145 * a482 + a15 * a539 + a181 * a473 + a181 * a522 + a265;
    TScalar a541 = a188 * a498;
    TScalar a542 = a13 * a539 + a193 * a473 + a193 * a522 + a20 * a531 + a270 + a272;
    TScalar a543 = (TScalar(1) / TScalar(2)) * a151;
    TScalar a544 = -a29 - a5 * a543;
    TScalar a545 = a155 * a52;
    TScalar a546 = a51 * a545;
    TScalar a547 = (TScalar(1) / TScalar(2)) * a125;
    TScalar a548 = -a19 - a4 * a547;
    TScalar a549 = a545 * a64;
    TScalar a550 = a23 * a543;
    TScalar a551 = a17 * a547;
    TScalar a552 = -a550 - a551;
    TScalar a553 = a545 * a80;
    TScalar a554 = -a186 - a4 * a543;
    TScalar a555 = a545 * a92;
    TScalar a556 = -a267 - a5 * a547;
    TScalar a557 = a103 * a545;
    TScalar a558 = a24 * a543;
    TScalar a559 = a0 * a547;
    TScalar a560 = -a558 - a559;
    TScalar a561 = a113 * a545;
    TScalar a562 = a128 * a545;
    TScalar a563 = a34 * a543;
    TScalar a564 = a154 * a32;
    TScalar a565 = a178 * a278;
    TScalar a566 = a139 * a545;
    TScalar a567 = a34 * a547;
    TScalar a568 = a125 * a81;
    TScalar a569 = a16 * a35;
    TScalar a570 = a32 * a569;
    TScalar a571 = a168 * a278;
    TScalar a572 = a21 * a543;
    TScalar a573 = a12 * a547;
    TScalar a574 = -a572 - a573;
    TScalar a575 = 2 * a157;
    TScalar a576 = a11 * a12;
    TScalar a577 = a152 * a545;
    TScalar a578 = a166 * a545;
    TScalar a579 = a157 * a35;
    TScalar a580 = a157 * a528 + a16 * a579 + a160 + a172 * a480 + a172 * a576 + a326;
    TScalar a581 = a176 * a545;
    TScalar a582 = a32 * a466;
    TScalar a583 = a11 * a157;
    TScalar a584 = a15 * a583 + a157 * a482 + a181 * a480 + a181 * a576 + a254 + a330;
    TScalar a585 = a20 * a543;
    TScalar a586 = a13 * a547;
    TScalar a587 = -a585 - a586;
    TScalar a588 = a188 * a545;
    TScalar a589 = a13 * a583 + a193 * a480 + a193 * a576 + a20 * a579 + a336;
    TScalar a590 = -a510 - a559;
    TScalar a591 = a170 * a52;
    TScalar a592 = a51 * a591;
    TScalar a593 = -a24 * a547 - a26;
    TScalar a594 = a591 * a64;
    TScalar a595 = -a23 * a501 - a29;
    TScalar a596 = a591 * a80;
    TScalar a597 = -a502 - a551;
    TScalar a598 = a591 * a92;
    TScalar a599 = -a174 - a23 * a547;
    TScalar a600 = a103 * a591;
    TScalar a601 = -a186 - a24 * a501;
    TScalar a602 = a113 * a591;
    TScalar a603 = -a536 - a586;
    TScalar a604 = a128 * a591;
    TScalar a605 = a139 * a591;
    TScalar a606 = a168 * a32;
    TScalar a607 = a152 * a591;
    TScalar a608 = -a520 - a573;
    TScalar a609 = 2 * a172;
    TScalar a610 = a166 * a591;
    TScalar a611 = a176 * a591;
    TScalar a612 = a11 * a172;
    TScalar a613 = a15 * a612 + a172 * a482 + a181 * a528 + a181 * a569;
    TScalar a614 = a188 * a591;
    TScalar a615 = a13 * a612 + a172 * a470 + a193 * a528 + a193 * a569;
    TScalar a616 = -a0 * a543 - a174;
    TScalar a617 = a179 * a52;
    TScalar a618 = a51 * a617;
    TScalar a619 = -a455 - a558;
    TScalar a620 = a617 * a64;
    TScalar a621 = -a17 * a445 - a19;
    TScalar a622 = a617 * a80;
    TScalar a623 = -a17 * a543 - a26;
    TScalar a624 = a617 * a92;
    TScalar a625 = -a446 - a550;
    TScalar a626 = a103 * a617;
    TScalar a627 = -a0 * a445 - a267;
    TScalar a628 = a113 * a617;
    TScalar a629 = a128 * a617;
    TScalar a630 = a154 * a199;
    TScalar a631 = a178 * a32;
    TScalar a632 = -a488 - a585;
    TScalar a633 = a139 * a617;
    TScalar a634 = a152 * a617;
    TScalar a635 = a131 * a199;
    TScalar a636 = a166 * a617;
    TScalar a637 = -a463 - a572;
    TScalar a638 = 2 * a181;
    TScalar a639 = a11 * a15;
    TScalar a640 = a176 * a617;
    TScalar a641 = a188 * a617;
    TScalar a642 = a11 * a13;
    TScalar a643 = a181 * a470 + a181 * a642 + a193 * a482 + a193 * a639;
    TScalar a644 = -a186 - a496 * a5;
    TScalar a645 = a191 * a52;
    TScalar a646 = a51 * a645;
    TScalar a647 = -a267 - a4 * a443;
    TScalar a648 = a64 * a645;
    TScalar a649 = -a454 - a509;
    TScalar a650 = a645 * a80;
    TScalar a651 = -a29 - a4 * a496;
    TScalar a652 = a645 * a92;
    TScalar a653 = -a19 - a443 * a5;
    TScalar a654 = a103 * a645;
    TScalar a655 = -a444 - a500;
    TScalar a656 = a113 * a645;
    TScalar a657 = a128 * a645;
    TScalar a658 = a141 * a278;
    TScalar a659 = a139 * a645;
    TScalar a660 = a130 * a278;
    TScalar a661 = -a487 - a535;
    TScalar a662 = a152 * a645;
    TScalar a663 = a166 * a645;
    TScalar a664 = a176 * a645;
    TScalar a665 = -a462 - a519;
    TScalar a666 = 2 * a193;
    TScalar a667 = a188 * a645;
    math::linalg::mini::SMatrix<TScalar, kDofs, kDofs> hess_d;
    hess_d[0] = a10 * a34 + a10 * a37 + a10 * a40 + a41 * a46 + a46 * a47 - 2 * a48 + a54 * a55 +
                a54 * a56 + a54 * a57;
    hess_d[1] = a11 * a19 * a44 * a52 * a64 + a26 * a38 * a44 * a52 * a64 +
                a30 * a35 * a44 * a52 * a64 - a34 * a58 - a37 * a58 - a59 * a60 - a74;
    hess_d[2] = a11 * a19 * a44 * a52 * a80 + a26 * a38 * a44 * a52 * a80 +
                a30 * a35 * a44 * a52 * a80 - a34 * a75 - a40 * a75 - a76 * a77 - a88;
    hess_d[3]  = a101 + a34 * a91 + a37 * a91 + a40 * a91 + a55 * a93 + a56 * a93 + a57 * a93;
    hess_d[4]  = a104 * a55 + a104 * a56 + a104 * a57 - a105 * a34 - a105 * a37 - a106 * a60 + a112;
    hess_d[5]  = a114 * a55 + a114 * a56 + a114 * a57 - a115 * a34 - a115 * a40 - a116 * a77 + a121;
    hess_d[6]  = a124 * a34 + a124 * a37 + a124 * a40 + a129 * a55 + a129 * a56 + a129 * a57 + a135;
    hess_d[7]  = a136 * a34 + a136 * a37 + a136 * a40 + a140 * a55 + a140 * a56 + a140 * a57 + a149;
    hess_d[8]  = a150 * a34 + a150 * a37 + a150 * a40 + a153 * a55 + a153 * a56 + a153 * a57 + a162;
    hess_d[9]  = a165 * a34 + a165 * a37 + a165 * a40 + a167 * a55 + a167 * a56 + a167 * a57 + a173;
    hess_d[10] = a175 * a34 + a175 * a37 + a175 * a40 + a177 * a55 + a177 * a56 + a177 * a57 + a185;
    hess_d[11] = a187 * a34 + a187 * a37 + a187 * a40 + a189 * a55 + a189 * a56 + a189 * a57 + a196;
    hess_d[12] = -a0 * a198 - a0 * a200 + a11 * a19 * a51 * a52 * a68 - a201 * a47 +
                 a26 * a38 * a51 * a52 * a68 + a30 * a35 * a51 * a52 * a68 - a74;
    hess_d[13] = a205 * a34 + a205 * a37 + a205 * a40 + a206 * a208 + a208 * a59 - 2 * a209 +
                 a211 * a55 + a211 * a56 + a211 * a57;
    hess_d[14] = a11 * a19 * a52 * a68 * a80 - a17 * a214 - a212 * a213 - a216 * a217 - a222 +
                 a26 * a38 * a52 * a68 * a80 + a30 * a35 * a52 * a68 * a80;
    hess_d[15] =
        -a17 * a198 - a17 * a200 - a201 * a215 + a223 * a55 + a223 * a56 + a223 * a57 + a226;
    hess_d[16] = a229 * a34 + a229 * a37 + a229 * a40 + a230 * a55 + a230 * a56 + a230 * a57 + a234;
    hess_d[17] =
        -a0 * a214 - a213 * a236 + a235 * a55 + a235 * a56 + a235 * a57 - a237 * a47 + a239;
    hess_d[18] = a240 * a34 + a240 * a37 + a240 * a40 + a241 * a55 + a241 * a56 + a241 * a57 + a243;
    hess_d[19] = a246 * a34 + a246 * a37 + a246 * a40 + a247 * a55 + a247 * a56 + a247 * a57 + a249;
    hess_d[20] = a250 * a34 + a250 * a37 + a250 * a40 + a251 * a55 + a251 * a56 + a251 * a57 + a256;
    hess_d[21] = a257 * a34 + a257 * a37 + a257 * a40 + a258 * a55 + a258 * a56 + a258 * a57 + a260;
    hess_d[22] = a263 * a34 + a263 * a37 + a263 * a40 + a264 * a55 + a264 * a56 + a264 * a57 + a266;
    hess_d[23] = a268 * a34 + a268 * a37 + a268 * a40 + a269 * a55 + a269 * a56 + a269 * a57 + a274;
    hess_d[24] = a11 * a19 * a51 * a52 * a83 + a26 * a38 * a51 * a52 * a83 - a276 * a5 -
                 a277 * a41 - a279 * a5 + a30 * a35 * a51 * a52 * a83 - a88;
    hess_d[25] = a11 * a19 * a52 * a64 * a83 - a222 + a26 * a38 * a52 * a64 * a83 - a281 * a4 -
                 a283 * a284 - a285 * a4 + a30 * a35 * a52 * a64 * a83;
    hess_d[26] = a212 * a290 + a288 * a34 + a288 * a37 + a288 * a40 + a290 * a76 - 2 * a291 +
                 a293 * a55 + a293 * a56 + a293 * a57;
    hess_d[27] = -a276 * a4 - a277 * a282 - a279 * a4 + a294 * a55 + a294 * a56 + a294 * a57 + a298;
    hess_d[28] = -a281 * a5 - a285 * a5 + a299 * a55 + a299 * a56 + a299 * a57 - a300 * a41 + a303;
    hess_d[29] = a306 * a34 + a306 * a37 + a306 * a40 + a307 * a55 + a307 * a56 + a307 * a57 + a309;
    hess_d[30] = a310 * a34 + a310 * a37 + a310 * a40 + a311 * a55 + a311 * a56 + a311 * a57 + a313;
    hess_d[31] = a314 * a34 + a314 * a37 + a314 * a40 + a315 * a55 + a315 * a56 + a315 * a57 + a317;
    hess_d[32] = a320 * a34 + a320 * a37 + a320 * a40 + a321 * a55 + a321 * a56 + a321 * a57 + a323;
    hess_d[33] = a324 * a34 + a324 * a37 + a324 * a40 + a325 * a55 + a325 * a56 + a325 * a57 + a327;
    hess_d[34] = a328 * a34 + a328 * a37 + a328 * a40 + a329 * a55 + a329 * a56 + a329 * a57 + a331;
    hess_d[35] = a334 * a34 + a334 * a37 + a334 * a40 + a335 * a55 + a335 * a56 + a335 * a57 + a337;
    hess_d[36] = a101 + a338 * a34 + a338 * a37 + a338 * a40 + a340 * a55 + a340 * a56 + a340 * a57;
    hess_d[37] =
        -a116 * a300 + a226 - a24 * a281 + a341 * a55 + a341 * a56 + a341 * a57 - a342 * a59;
    hess_d[38] =
        -a106 * a237 - a23 * a344 + a298 + a343 * a55 + a343 * a56 + a343 * a57 - a345 * a76;
    hess_d[39] = 2 * a17 * a347 + 2 * a282 * a99 + a34 * a346 + a346 * a37 + a346 * a40 +
                 a348 * a55 + a348 * a56 + a348 * a57;
    hess_d[40] =
        -a106 * a342 - a23 * a281 - a300 * a76 + a349 * a55 + a349 * a56 + a349 * a57 + a351;
    hess_d[41] =
        -a116 * a345 - a237 * a59 - a24 * a344 + a352 * a55 + a352 * a56 + a352 * a57 + a354;
    hess_d[42] = a34 * a355 + a355 * a37 + a355 * a40 + a356 * a55 + a356 * a56 + a356 * a57 + a357;
    hess_d[43] = a34 * a358 + a358 * a37 + a358 * a40 + a359 * a55 + a359 * a56 + a359 * a57 + a360;
    hess_d[44] = a34 * a361 + a361 * a37 + a361 * a40 + a362 * a55 + a362 * a56 + a362 * a57 + a363;
    hess_d[45] = a34 * a364 + a364 * a37 + a364 * a40 + a365 * a55 + a365 * a56 + a365 * a57 + a366;
    hess_d[46] = a34 * a367 + a367 * a37 + a367 * a40 + a368 * a55 + a368 * a56 + a368 * a57 + a369;
    hess_d[47] = a34 * a370 + a37 * a370 + a370 * a40 + a371 * a55 + a371 * a56 + a371 * a57 + a372;
    hess_d[48] =
        -a0 * a276 - a0 * a375 - a107 * a32 * a47 + a112 + a374 * a55 + a374 * a56 + a374 * a57;
    hess_d[49] = a234 + a34 * a376 + a37 * a376 + a376 * a40 + a377 * a55 + a377 * a56 + a377 * a57;
    hess_d[50] =
        -a17 * a380 + a303 - a34 * a379 + a378 * a55 + a378 * a56 + a378 * a57 - a379 * a40;
    hess_d[51] =
        -a107 * a216 - a17 * a276 - a17 * a375 + a351 + a381 * a55 + a381 * a56 + a381 * a57;
    hess_d[52] = 2 * a106 * a111 + a34 * a382 + a37 * a382 + a382 * a40 + 2 * a383 * a5 +
                 a384 * a55 + a384 * a56 + a384 * a57;
    hess_d[53] = -a0 * a380 - a34 * a386 + a385 * a55 + a385 * a56 + a385 * a57 - a386 * a40 + a389;
    hess_d[54] = a34 * a390 + a37 * a390 + a390 * a40 + a391 * a55 + a391 * a56 + a391 * a57 + a393;
    hess_d[55] = a34 * a394 + a37 * a394 + a394 * a40 + a395 * a55 + a395 * a56 + a395 * a57 + a396;
    hess_d[56] = a34 * a397 + a37 * a397 + a397 * a40 + a398 * a55 + a398 * a56 + a398 * a57 + a399;
    hess_d[57] = a34 * a400 + a37 * a400 + a40 * a400 + a401 * a55 + a401 * a56 + a401 * a57 + a402;
    hess_d[58] = a34 * a403 + a37 * a403 + a40 * a403 + a404 * a55 + a404 * a56 + a404 * a57 + a405;
    hess_d[59] = a34 * a406 + a37 * a406 + a40 * a406 + a407 * a55 + a407 * a56 + a407 * a57 + a408;
    hess_d[60] =
        -a117 * a32 * a41 + a121 - a198 * a5 + a410 * a55 + a410 * a56 + a410 * a57 - a411 * a5;
    hess_d[61] = a239 - a34 * a413 - a37 * a413 - a4 * a414 + a412 * a55 + a412 * a56 + a412 * a57;
    hess_d[62] = a309 + a34 * a415 + a37 * a415 + a40 * a415 + a416 * a55 + a416 * a56 + a416 * a57;
    hess_d[63] = -a117 * a283 - a198 * a4 + a354 - a4 * a411 + a417 * a55 + a417 * a56 + a417 * a57;
    hess_d[64] = -a34 * a419 - a37 * a419 + a389 - a414 * a5 + a418 * a55 + a418 * a56 + a418 * a57;
    hess_d[65] = 2 * a120 * a236 + 2 * a24 * a421 + a34 * a420 + a37 * a420 + a40 * a420 +
                 a422 * a55 + a422 * a56 + a422 * a57;
    hess_d[66] = a34 * a423 + a37 * a423 + a40 * a423 + a424 * a55 + a424 * a56 + a424 * a57 + a426;
    hess_d[67] = a34 * a427 + a37 * a427 + a40 * a427 + a428 * a55 + a428 * a56 + a428 * a57 + a430;
    hess_d[68] = a34 * a431 + a37 * a431 + a40 * a431 + a432 * a55 + a432 * a56 + a432 * a57 + a433;
    hess_d[69] = a34 * a434 + a37 * a434 + a40 * a434 + a435 * a55 + a435 * a56 + a435 * a57 + a436;
    hess_d[70] = a34 * a437 + a37 * a437 + a40 * a437 + a438 * a55 + a438 * a56 + a438 * a57 + a439;
    hess_d[71] = a34 * a440 + a37 * a440 + a40 * a440 + a441 * a55 + a441 * a56 + a441 * a57 + a442;
    hess_d[72] = a135 + a34 * a447 + a37 * a447 + a40 * a447 + a449 * a55 + a449 * a56 + a449 * a57;
    hess_d[73] = a243 + a34 * a450 + a37 * a450 + a40 * a450 + a451 * a55 + a451 * a56 + a451 * a57;
    hess_d[74] = a313 + a34 * a452 + a37 * a452 + a40 * a452 + a453 * a55 + a453 * a56 + a453 * a57;
    hess_d[75] = a34 * a456 + a357 + a37 * a456 + a40 * a456 + a457 * a55 + a457 * a56 + a457 * a57;
    hess_d[76] = a34 * a458 + a37 * a458 + a393 + a40 * a458 + a459 * a55 + a459 * a56 + a459 * a57;
    hess_d[77] = a34 * a460 + a37 * a460 + a40 * a460 + a426 + a461 * a55 + a461 * a56 + a461 * a57;
    hess_d[78] = 2 * a13 * a465 + 2 * a133 + 2 * a134 * a466 + a34 * a464 + a37 * a464 +
                 a40 * a464 + a467 * a55 + a467 * a56 + a467 * a57;
    hess_d[79] =
        -a20 * a469 + a468 * a55 + a468 * a56 + a468 * a57 - a471 * a472 - a473 * a474 + a477;
    hess_d[80] =
        -a131 * a481 - a21 * a479 + a478 * a55 + a478 * a56 + a478 * a57 - a482 * a484 + a486;
    hess_d[81] = a34 * a489 + a37 * a489 + a40 * a489 + a490 * a55 + a490 * a56 + a490 * a57 + a491;
    hess_d[82] =
        -a21 * a469 - a472 * a481 - a474 * a482 + a492 * a55 + a492 * a56 + a492 * a57 + a493;
    hess_d[83] =
        -a131 * a471 - a20 * a479 - a473 * a484 + a494 * a55 + a494 * a56 + a494 * a57 + a495;
    hess_d[84] = a149 + a34 * a497 + a37 * a497 + a40 * a497 + a499 * a55 + a499 * a56 + a499 * a57;
    hess_d[85] = a249 + a34 * a503 + a37 * a503 + a40 * a503 + a504 * a55 + a504 * a56 + a504 * a57;
    hess_d[86] = a317 + a34 * a505 + a37 * a505 + a40 * a505 + a506 * a55 + a506 * a56 + a506 * a57;
    hess_d[87] = a34 * a507 + a360 + a37 * a507 + a40 * a507 + a508 * a55 + a508 * a56 + a508 * a57;
    hess_d[88] = a34 * a511 + a37 * a511 + a396 + a40 * a511 + a512 * a55 + a512 * a56 + a512 * a57;
    hess_d[89] = a34 * a513 + a37 * a513 + a40 * a513 + a430 + a514 * a55 + a514 * a56 + a514 * a57;
    hess_d[90] =
        -a13 * a516 - a13 * a517 - a475 * a518 + a477 + a515 * a55 + a515 * a56 + a515 * a57;
    hess_d[91] = 2 * a248 + a34 * a521 + a37 * a521 + a40 * a521 + a473 * a523 + a522 * a523 +
                 a524 * a55 + a524 * a56 + a524 * a57;
    hess_d[92] =
        -a12 * a526 - a12 * a527 + a525 * a55 + a525 * a56 + a525 * a57 - a528 * a529 + a532;
    hess_d[93] =
        -a12 * a516 - a12 * a517 - a518 * a528 + a533 * a55 + a533 * a56 + a533 * a57 + a534;
    hess_d[94] = a34 * a537 + a37 * a537 + a40 * a537 + a538 * a55 + a538 * a56 + a538 * a57 + a540;
    hess_d[95] =
        -a13 * a526 - a13 * a527 - a475 * a529 + a541 * a55 + a541 * a56 + a541 * a57 + a542;
    hess_d[96] = a162 + a34 * a544 + a37 * a544 + a40 * a544 + a546 * a55 + a546 * a56 + a546 * a57;
    hess_d[97] = a256 + a34 * a548 + a37 * a548 + a40 * a548 + a549 * a55 + a549 * a56 + a549 * a57;
    hess_d[98] = a323 + a34 * a552 + a37 * a552 + a40 * a552 + a55 * a553 + a553 * a56 + a553 * a57;
    hess_d[99] = a34 * a554 + a363 + a37 * a554 + a40 * a554 + a55 * a555 + a555 * a56 + a555 * a57;
    hess_d[100] =
        a34 * a556 + a37 * a556 + a399 + a40 * a556 + a55 * a557 + a557 * a56 + a557 * a57;
    hess_d[101] =
        a34 * a560 + a37 * a560 + a40 * a560 + a433 + a55 * a561 + a56 * a561 + a561 * a57;
    hess_d[102] =
        -a15 * a563 - a15 * a565 - a466 * a564 + a486 + a55 * a562 + a56 * a562 + a562 * a57;
    hess_d[103] =
        -a16 * a567 - a16 * a571 + a532 + a55 * a566 + a56 * a566 + a566 * a57 - a568 * a570;
    hess_d[104] = 2 * a322 + a34 * a574 + a37 * a574 + a40 * a574 + a480 * a575 + a55 * a577 +
                  a56 * a577 + a57 * a577 + a575 * a576;
    hess_d[105] =
        -a16 * a563 - a16 * a565 + a55 * a578 + a56 * a578 - a564 * a569 + a57 * a578 + a580;
    hess_d[106] =
        -a15 * a567 - a15 * a571 + a55 * a581 + a56 * a581 - a568 * a582 + a57 * a581 + a584;
    hess_d[107] =
        a34 * a587 + a37 * a587 + a40 * a587 + a55 * a588 + a56 * a588 + a57 * a588 + a589;
    hess_d[108] =
        a173 + a34 * a590 + a37 * a590 + a40 * a590 + a55 * a592 + a56 * a592 + a57 * a592;
    hess_d[109] =
        a260 + a34 * a593 + a37 * a593 + a40 * a593 + a55 * a594 + a56 * a594 + a57 * a594;
    hess_d[110] =
        a327 + a34 * a595 + a37 * a595 + a40 * a595 + a55 * a596 + a56 * a596 + a57 * a596;
    hess_d[111] =
        a34 * a597 + a366 + a37 * a597 + a40 * a597 + a55 * a598 + a56 * a598 + a57 * a598;
    hess_d[112] =
        a34 * a599 + a37 * a599 + a40 * a599 + a402 + a55 * a600 + a56 * a600 + a57 * a600;
    hess_d[113] =
        a34 * a601 + a37 * a601 + a40 * a601 + a436 + a55 * a602 + a56 * a602 + a57 * a602;
    hess_d[114] =
        a34 * a603 + a37 * a603 + a40 * a603 + a491 + a55 * a604 + a56 * a604 + a57 * a604;
    hess_d[115] =
        -a20 * a567 - a471 * a568 - a473 * a606 + a534 + a55 * a605 + a56 * a605 + a57 * a605;
    hess_d[116] =
        -a169 * a481 - a21 * a526 - a482 * a529 + a55 * a607 + a56 * a607 + a57 * a607 + a580;
    hess_d[117] = a34 * a608 + a37 * a608 + a40 * a608 + a528 * a609 + a55 * a610 + a56 * a610 +
                  a569 * a609 + a57 * a610;
    hess_d[118] =
        -a21 * a567 - a481 * a568 - a482 * a606 + a55 * a611 + a56 * a611 + a57 * a611 + a613;
    hess_d[119] =
        -a169 * a471 - a20 * a526 - a473 * a529 + a55 * a614 + a56 * a614 + a57 * a614 + a615;
    hess_d[120] =
        a185 + a34 * a616 + a37 * a616 + a40 * a616 + a55 * a618 + a56 * a618 + a57 * a618;
    hess_d[121] =
        a266 + a34 * a619 + a37 * a619 + a40 * a619 + a55 * a620 + a56 * a620 + a57 * a620;
    hess_d[122] =
        a331 + a34 * a621 + a37 * a621 + a40 * a621 + a55 * a622 + a56 * a622 + a57 * a622;
    hess_d[123] =
        a34 * a623 + a369 + a37 * a623 + a40 * a623 + a55 * a624 + a56 * a624 + a57 * a624;
    hess_d[124] =
        a34 * a625 + a37 * a625 + a40 * a625 + a405 + a55 * a626 + a56 * a626 + a57 * a626;
    hess_d[125] =
        a34 * a627 + a37 * a627 + a40 * a627 + a439 + a55 * a628 + a56 * a628 + a57 * a628;
    hess_d[126] =
        -a13 * a563 - a13 * a630 - a475 * a631 + a493 + a55 * a629 + a56 * a629 + a57 * a629;
    hess_d[127] =
        a34 * a632 + a37 * a632 + a40 * a632 + a540 + a55 * a633 + a56 * a633 + a57 * a633;
    hess_d[128] =
        -a12 * a479 - a12 * a635 - a484 * a528 + a55 * a634 + a56 * a634 + a57 * a634 + a584;
    hess_d[129] =
        -a12 * a563 - a12 * a630 - a528 * a631 + a55 * a636 + a56 * a636 + a57 * a636 + a613;
    hess_d[130] = a34 * a637 + a37 * a637 + a40 * a637 + a482 * a638 + a55 * a640 + a56 * a640 +
                  a57 * a640 + a638 * a639;
    hess_d[131] =
        -a13 * a479 - a13 * a635 - a475 * a484 + a55 * a641 + a56 * a641 + a57 * a641 + a643;
    hess_d[132] =
        a196 + a34 * a644 + a37 * a644 + a40 * a644 + a55 * a646 + a56 * a646 + a57 * a646;
    hess_d[133] =
        a274 + a34 * a647 + a37 * a647 + a40 * a647 + a55 * a648 + a56 * a648 + a57 * a648;
    hess_d[134] =
        a337 + a34 * a649 + a37 * a649 + a40 * a649 + a55 * a650 + a56 * a650 + a57 * a650;
    hess_d[135] =
        a34 * a651 + a37 * a651 + a372 + a40 * a651 + a55 * a652 + a56 * a652 + a57 * a652;
    hess_d[136] =
        a34 * a653 + a37 * a653 + a40 * a653 + a408 + a55 * a654 + a56 * a654 + a57 * a654;
    hess_d[137] =
        a34 * a655 + a37 * a655 + a40 * a655 + a442 + a55 * a656 + a56 * a656 + a57 * a656;
    hess_d[138] =
        -a15 * a516 - a15 * a658 - a190 * a582 + a495 + a55 * a657 + a56 * a657 + a57 * a657;
    hess_d[139] =
        -a16 * a469 - a16 * a660 - a472 * a570 + a542 + a55 * a659 + a56 * a659 + a57 * a659;
    hess_d[140] =
        a34 * a661 + a37 * a661 + a40 * a661 + a55 * a662 + a56 * a662 + a57 * a662 + a589;
    hess_d[141] =
        -a16 * a516 - a16 * a658 - a190 * a570 + a55 * a663 + a56 * a663 + a57 * a663 + a615;
    hess_d[142] =
        -a15 * a469 - a15 * a660 - a472 * a582 + a55 * a664 + a56 * a664 + a57 * a664 + a643;
    hess_d[143] = a34 * a665 + a37 * a665 + a40 * a665 + a470 * a666 + a55 * a667 + a56 * a667 +
                  a57 * a667 + a642 * a666;
    return hess_d;
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_MESHDISTANCE_H
