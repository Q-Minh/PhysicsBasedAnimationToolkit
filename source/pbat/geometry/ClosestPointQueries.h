#ifndef PBAT_GEOMETRY_CLOSEST_POINT_QUERIES_H
#define PBAT_GEOMETRY_CLOSEST_POINT_QUERIES_H

#include <Eigen/Geometry>
#include <pbat/Aliases.h>

namespace pbat {
namespace geometry {
namespace ClosestPointQueries {

/**
 * @brief Obtain the point on the plane (P,n) closest to the point X.
 * @param X
 * @param P
 * @param n
 * @return
 */
template <class TDerivedX, class TDerivedP, class TDerivedN>
Vector<TDerivedX::RowsAtCompileTime> PointOnPlane(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedN> const& n);

/**
 * @brief Obtain the point on the line segment PQ closest to the point X.
 * @param X
 * @param P
 * @param Q
 * @return
 */
template <class TDerivedX, class TDerivedP, class TDerivedQ>
Vector<TDerivedX::RowsAtCompileTime> PointOnLineSegment(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q);

/**
 * @brief Obtain the point on the axis-aligned bounding box (AABB) defined by the lower
 * and upper corners closest to the point P.
 * @param P
 * @param L
 * @param U
 * @return
 */
template <class TDerivedX, class TDerivedL, class TDerivedU>
Vector<TDerivedX::RowsAtCompileTime> PointOnAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U);

/**
 * @brief Obtain the point on the triangle ABC closest to the point P in barycentric coordinates.
 * @param P
 * @param A
 * @param B
 * @param C
 * @return
 */
template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC>
Vector<3> UvwPointInTriangle(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C);

/**
 * @brief Obtain the point on the triangle ABC closest to the point P.
 * @param P
 * @param A
 * @param B
 * @param C
 * @return
 */
template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC>
Vector<TDerivedP::RowsAtCompileTime> PointInTriangle(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C);

/**
 * @brief Obtain the point in the tetrahedron ABCD closest to the point P. The order of ABCD
 * must be such that all faces ABC, ACD, ADB and BDC are oriented with outwards pointing normals
 * when viewed from outside the tetrahedron.
 * @param P
 * @param A
 * @param B
 * @param C
 * @param D
 * @return
 */
template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedD>
Vector<TDerivedP::RowsAtCompileTime> PointInTetrahedron(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedD> const& D);

template <class TDerivedX, class TDerivedP, class TDerivedN>
Vector<TDerivedX::RowsAtCompileTime> PointOnPlane(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedN> const& n)
{
#ifndef NDEBUG
    bool const bIsNormalUnit = std::abs(n.squaredNorm() - 1.) <= 1e-15;
    assert(bIsNormalUnit);
#endif
    /**
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.1.1
     */
    Scalar const t    = n.dot(X - P);
    auto const Xplane = X - t * n;
    return Xplane;
}

template <class TDerivedX, class TDerivedP, class TDerivedQ>
Vector<TDerivedX::RowsAtCompileTime> PointOnLineSegment(
    Eigen::MatrixBase<TDerivedX> const& X,
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedQ> const& Q)
{
    /**
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.1.2
     */
    Vector<TDerivedX::RowsAtCompileTime> const PQ = Q - P;
    // Project X onto PQ, computing parameterized position R(t) = P + t*(Q � P)
    Scalar t = (X - P).dot(PQ) / PQ.squaredNorm();
    // If outside segment, clamp t (and therefore d) to the closest endpoint
    t = std::clamp(t, 0., 1.);
    // Compute projected position from the clamped t
    auto const Xpq = P + t * PQ;
    return Xpq;
}

template <class TDerivedX, class TDerivedL, class TDerivedU>
Vector<TDerivedX::RowsAtCompileTime> PointOnAxisAlignedBoundingBox(
    Eigen::MatrixBase<TDerivedX> const& P,
    Eigen::MatrixBase<TDerivedL> const& L,
    Eigen::MatrixBase<TDerivedU> const& U)
{
    /**
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.1.3
     */
    Vector<TDerivedX::RowsAtCompileTime> X = P;
    for (auto i = 0; i < P.rows(); ++i)
        X(i) = std::clamp(X(i), L(i), U(i));
    return X;
}

template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC>
Vector<3> UvwPointInTriangle(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C)
{
    /**
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.1.5
     */

    // Check if P in vertex region outside A
    Vector<TDerivedP::RowsAtCompileTime> const AB = B - A;
    Vector<TDerivedP::RowsAtCompileTime> const AC = C - A;
    Vector<TDerivedP::RowsAtCompileTime> const AP = P - A;
    Scalar const d1                               = AB.dot(AP);
    Scalar const d2                               = AC.dot(AP);
    if (d1 <= 0. and d2 <= 0.)
        return Vector<3>::UnitX(); // barycentric coordinates (1,0,0)

    // Check if P in vertex region outside B
    Vector<TDerivedP::RowsAtCompileTime> const BP = P - B;
    Scalar const d3                               = AB.dot(BP);
    Scalar const d4                               = AC.dot(BP);
    if (d3 >= 0. and d4 <= d3)
        return Vector<3>::UnitY(); // barycentric coordinates (0,1,0)

    // Check if P in edge region of AB, if so return projection of P onto AB
    Scalar const vc = d1 * d4 - d3 * d2;
    if (vc <= 0. and d1 >= 0. and d3 <= 0.)
    {
        Scalar const v = d1 / (d1 - d3);
        return Vector<3>{1. - v, v, 0.}; // barycentric coordinates (1-v,v,0)
    }

    // Check if P in vertex region outside C
    Vector<TDerivedP::RowsAtCompileTime> const CP = P - C;
    Scalar const d5                               = AB.dot(CP);
    Scalar const d6                               = AC.dot(CP);
    if (d6 >= 0. and d5 <= d6)
        return Vector<3>::UnitZ(); // barycentric coordinates (0,0,1)

    // Check if P in edge region of AC, if so return projection of P onto AC
    Scalar const vb = d5 * d2 - d1 * d6;
    if (vb <= 0. and d2 >= 0. and d6 <= 0.)
    {
        Scalar const w = d2 / (d2 - d6);
        return Vector<3>{1. - w, 0., w}; // barycentric coordinates (1-w,0,w)
    }
    // Check if P in edge region of BC, if so return projection of P onto BC
    Scalar const va = d3 * d6 - d5 * d4;
    if (va <= 0. and (d4 - d3) >= 0. and (d5 - d6) >= 0.)
    {
        Scalar const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Vector<3>{0., 1. - w, w}; // barycentric coordinates (0,1-w,w)
    }
    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    Scalar const denom = 1. / (va + vb + vc);
    Scalar const v     = vb * denom;
    Scalar const w     = vc * denom;
    return Vector<3>{1. - v - w, v, w}; // = u*a + v*b + w*c, u = va * denom = 1.0f-v-w
}

template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC>
Vector<TDerivedP::RowsAtCompileTime> PointInTriangle(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C)
{
    Vector<3> const uvw = UvwPointInTriangle(P, A, B, C);
    return A * uvw(0) + B * uvw(1) + C * uvw(2);
}

template <class TDerivedP, class TDerivedA, class TDerivedB, class TDerivedC, class TDerivedD>
Vector<TDerivedP::RowsAtCompileTime> PointInTetrahedron(
    Eigen::MatrixBase<TDerivedP> const& P,
    Eigen::MatrixBase<TDerivedA> const& A,
    Eigen::MatrixBase<TDerivedB> const& B,
    Eigen::MatrixBase<TDerivedC> const& C,
    Eigen::MatrixBase<TDerivedD> const& D)
{
    /**
     * Ericson, Christer. Real-time collision detection. Crc Press, 2004. section 5.1.6
     */

    // Start out assuming point inside all halfspaces, so closest to itself
    Vector<TDerivedP::RowsAtCompileTime> X = P;
    Scalar d2min                           = std::numeric_limits<Scalar>::max();

    auto const PointOutsidePlane = [](auto const& p, auto const& a, auto const& b, auto const& c) {
        Scalar const d = (p - a).dot((b - a).cross(c - a));
        return d > 0.;
    };
    auto const TestFace = [&](auto const& a, auto const& b, auto const& c) {
        // If point outside face abc then compute closest point on abc
        if (PointOutsidePlane(P, a, b, c))
        {
            auto const Q    = PointInTriangle(P, a, b, c);
            Scalar const d2 = (Q - P).squaredNorm();
            // Update best closest point if (squared) distance is less than current best
            if (d2 < d2min)
            {
                d2min = d2;
                X     = Q;
            }
        }
    };
    TestFace(A, B, D);
    TestFace(B, C, D);
    TestFace(C, A, D);
    TestFace(A, C, B);
    return X;
}

} // namespace ClosestPointQueries
} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_CLOSEST_POINT_QUERIES_H