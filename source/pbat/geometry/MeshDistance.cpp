#include "MeshDistance.h"

#include "pbat/Aliases.h"

#include <doctest/doctest.h>
#include <limits>

namespace pbat::geometry::test {

template <class TScalar>
math::linalg::mini::SVector<TScalar, 6>
SymbolicPointPointGradient(math::linalg::mini::SVector<TScalar, 6> const& x);

template <class TScalar>
math::linalg::mini::SVector<TScalar, 9>
SymbolicPointEdgeGradient(math::linalg::mini::SVector<TScalar, 9> const& x);

template <class TScalar>
math::linalg::mini::SVector<TScalar, 12>
SymbolicPointTriangleGradient(math::linalg::mini::SVector<TScalar, 12> const& x);

template <class TScalar>
math::linalg::mini::SVector<TScalar, 12> SymbolicEdgeEdgeGradient(
    math::linalg::mini::SVector<TScalar, 12> const& x,
    TScalar eps = std::numeric_limits<TScalar>::epsilon());

} // namespace pbat::geometry::test

TEST_CASE("[geometry] Point-point distance can be computed")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::geometry;
    using ScalarType = pbat::Scalar;

    SUBCASE("Coincident points")
    {
        // Arrange
        SVector<ScalarType, 6> x;
        x(0) = 1.;
        x(1) = 2.;
        x(2) = 3.;
        x(3) = 1.;
        x(4) = 2.;
        x(5) = 3.;
        // Act
        ScalarType const d = PointPointDistance<ScalarType>{}.Eval(x);
        // Assert
        CHECK_EQ(d, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Unit distance")
    {
        // Arrange
        SVector<ScalarType, 6> x;
        x(0) = 0.;
        x(1) = 0.;
        x(2) = 0.;
        x(3) = 1.;
        x(4) = 0.;
        x(5) = 0.;
        // Act
        ScalarType const d = PointPointDistance<ScalarType>{}.Eval(x);
        auto g             = PointPointDistance<ScalarType>{}.Gradient(x);
        auto gsym          = pbat::geometry::test::SymbolicPointPointGradient(x);
        ScalarType gerror  = SquaredNorm(g - gsym);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-10));
        CHECK_EQ(gerror, doctest::Approx(0.).epsilon(1e-10));
    }
}

TEST_CASE("[geometry] Point-edge distance can be computed")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::geometry;
    using ScalarType = pbat::Scalar;

    SUBCASE("Point on edge")
    {
        // Arrange
        // Edge from (0,0,0) to (2,0,0), point at (1,0,0)
        SVector<ScalarType, 9> x;
        x(0) = 1.;
        x(1) = 0.;
        x(2) = 0.; // x
        x(3) = 0.;
        x(4) = 0.;
        x(5) = 0.; // a
        x(6) = 2.;
        x(7) = 0.;
        x(8) = 0.; // b
        // Act
        ScalarType const d = PointEdgeDistance<ScalarType>{}.Eval(x);
        // Assert
        CHECK_EQ(d, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Point perpendicular to edge")
    {
        // Arrange
        // Edge from (0,0,0) to (2,0,0), point at (1,1,0)
        SVector<ScalarType, 9> x;
        x(0) = 1.;
        x(1) = 1.;
        x(2) = 0.; // x
        x(3) = 0.;
        x(4) = 0.;
        x(5) = 0.; // a
        x(6) = 2.;
        x(7) = 0.;
        x(8) = 0.; // b
        // Act
        ScalarType const d = PointEdgeDistance<ScalarType>{}.Eval(x);
        auto g             = PointEdgeDistance<ScalarType>{}.Gradient(x);
        auto gsym          = pbat::geometry::test::SymbolicPointEdgeGradient(x);
        ScalarType gerror  = SquaredNorm(g - gsym);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-10));
        CHECK_EQ(gerror, doctest::Approx(0.).epsilon(1e-10));
    }
}

TEST_CASE("[geometry] Point-triangle distance can be computed")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::geometry;
    using ScalarType = pbat::Scalar;

    SUBCASE("Point on triangle plane")
    {
        // Arrange
        // Triangle with vertices at (0,0,0), (1,0,0), (0,1,0)
        // Point at (0.25, 0.25, 0)
        SVector<ScalarType, 12> x;
        x(0)  = 0.25;
        x(1)  = 0.25;
        x(2)  = 0.; // x
        x(3)  = 0.;
        x(4)  = 0.;
        x(5)  = 0.; // a
        x(6)  = 1.;
        x(7)  = 0.;
        x(8)  = 0.; // b
        x(9)  = 0.;
        x(10) = 1.;
        x(11) = 0.; // c
        // Act
        ScalarType const d = PointTriangleDistance<ScalarType>{}.Eval(x);
        auto g             = PointTriangleDistance<ScalarType>{}.Gradient(x);
        auto gsym          = pbat::geometry::test::SymbolicPointTriangleGradient(x);
        ScalarType gerror  = SquaredNorm(g - gsym);
        // Assert
        CHECK_EQ(d, doctest::Approx(0.).epsilon(1e-10));
        CHECK_EQ(gerror, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Point above triangle")
    {
        // Arrange
        // Triangle with vertices at (0,0,0), (1,0,0), (0,1,0)
        // Point at (0.25, 0.25, 1)
        SVector<ScalarType, 12> x;
        x(0)  = 0.25;
        x(1)  = 0.25;
        x(2)  = 1.; // x
        x(3)  = 0.;
        x(4)  = 0.;
        x(5)  = 0.; // a
        x(6)  = 1.;
        x(7)  = 0.;
        x(8)  = 0.; // b
        x(9)  = 0.;
        x(10) = 1.;
        x(11) = 0.; // c
        // Act
        ScalarType const d = PointTriangleDistance<ScalarType>{}.Eval(x);
        auto g             = PointTriangleDistance<ScalarType>{}.Gradient(x);
        auto gsym          = pbat::geometry::test::SymbolicPointTriangleGradient(x);
        ScalarType gerror  = SquaredNorm(g - gsym);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-10));
        CHECK_EQ(gerror, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Point below triangle")
    {
        // Arrange
        // Triangle with vertices at (0,0,0), (1,0,0), (0,1,0)
        // Point at (0.25, 0.25, -1)
        SVector<ScalarType, 12> x;
        x(0)  = 0.25;
        x(1)  = 0.25;
        x(2)  = -1.; // x
        x(3)  = 0.;
        x(4)  = 0.;
        x(5)  = 0.; // a
        x(6)  = 1.;
        x(7)  = 0.;
        x(8)  = 0.; // b
        x(9)  = 0.;
        x(10) = 1.;
        x(11) = 0.; // c
        // Act
        ScalarType const d = PointTriangleDistance<ScalarType>{}.Eval(x);
        auto g             = PointTriangleDistance<ScalarType>{}.Gradient(x);
        auto gsym          = pbat::geometry::test::SymbolicPointTriangleGradient(x);
        ScalarType gerror  = SquaredNorm(g - gsym);
        // Assert
        CHECK_EQ(d, doctest::Approx(-1.).epsilon(1e-10));
        CHECK_EQ(gerror, doctest::Approx(0.).epsilon(1e-10));
    }
}

TEST_CASE("[geometry] Edge-edge distance can be computed")
{
    using namespace pbat::math::linalg::mini;
    using namespace pbat::geometry;
    using ScalarType = pbat::Scalar;

    SUBCASE("Intersecting edges")
    {
        // Arrange
        // Edge 1 from (-1,0,0) to (1,0,0)
        // Edge 2 from (0,-1,0) to (0,1,0)
        SVector<ScalarType, 12> x;
        x(0)                 = -1.;
        x(1)                 = 0.;
        x(2)                 = 0.; // a
        x(3)                 = 1.;
        x(4)                 = 0.;
        x(5)                 = 0.; // b
        x(6)                 = 0.;
        x(7)                 = -1.;
        x(8)                 = 0.; // c
        x(9)                 = 0.;
        x(10)                = 1.;
        x(11)                = 0.; // d
        ScalarType const eps = 1e-6;
        // Act
        ScalarType const d = EdgeEdgeDistance<ScalarType>{}.Eval(x, eps);
        auto g             = EdgeEdgeDistance<ScalarType>{}.Gradient(x, eps);
        auto gsym          = pbat::geometry::test::SymbolicEdgeEdgeGradient(x, eps);
        ScalarType gerror  = SquaredNorm(g - gsym);
        // Assert
        CHECK_EQ(d, doctest::Approx(0.).epsilon(1e-5));
        CHECK_EQ(gerror, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Parallel edges")
    {
        // Arrange
        // Edge 1 from (0,0,0) to (1,0,0)
        // Edge 2 from (0,1,0) to (1,1,0)
        SVector<ScalarType, 12> x;
        x(0)                 = 0.;
        x(1)                 = 0.;
        x(2)                 = 0.; // a
        x(3)                 = 1.;
        x(4)                 = 0.;
        x(5)                 = 0.; // b
        x(6)                 = 0.;
        x(7)                 = 1.;
        x(8)                 = 0.; // c
        x(9)                 = 1.;
        x(10)                = 1.;
        x(11)                = 0.; // d
        ScalarType const eps = 1e-6;
        // Act
        ScalarType const d = EdgeEdgeDistance<ScalarType>{}.Eval(x, eps);
        auto g             = EdgeEdgeDistance<ScalarType>{}.Gradient(x, eps);
        auto gsym          = pbat::geometry::test::SymbolicEdgeEdgeGradient(x, eps);
        ScalarType gerror  = SquaredNorm(g - gsym);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-5));
        CHECK_EQ(gerror, doctest::Approx(0.).epsilon(1e-10));
    }
    SUBCASE("Non-intersecting edges")
    {
        // Arrange
        // Edge 1 from (0,0,0) to (1,0,0)
        // Edge 2 from (0.5,1,0) to (0.5,1,1)
        SVector<ScalarType, 12> x;
        x(0)                 = 0.;
        x(1)                 = 0.;
        x(2)                 = 0.; // a
        x(3)                 = 1.;
        x(4)                 = 0.;
        x(5)                 = 0.; // b
        x(6)                 = 0.5;
        x(7)                 = 1.;
        x(8)                 = 0.; // c
        x(9)                 = 0.5;
        x(10)                = 1.;
        x(11)                = 1.; // d
        ScalarType const eps = 1e-6;
        // Act
        ScalarType const d = EdgeEdgeDistance<ScalarType>{}.Eval(x, eps);
        auto g             = EdgeEdgeDistance<ScalarType>{}.Gradient(x, eps);
        auto gsym          = pbat::geometry::test::SymbolicEdgeEdgeGradient(x);
        ScalarType gerror  = SquaredNorm(g - gsym);
        // Assert
        CHECK_EQ(d, doctest::Approx(1.).epsilon(1e-5));
        CHECK_EQ(gerror, doctest::Approx(0.).epsilon(1e-10));
    }
}

namespace pbat::geometry::test {
template <class TScalar>
math::linalg::mini::SVector<TScalar, 6>
SymbolicPointPointGradient(math::linalg::mini::SVector<TScalar, 6> const& x)
{
    auto x_0 = x(0);
    auto x_1 = x(1);
    auto x_2 = x(2);
    auto y_0 = x(3);
    auto y_1 = x(4);
    auto y_2 = x(5);
    math::linalg::mini::SVector<TScalar, 6> grad_d;
    TScalar a0 = x_0 - y_0;
    TScalar a1 = x_1 - y_1;
    TScalar a2 = x_2 - y_2;
    TScalar a3 = 1 / std::sqrt(((a0) * (a0)) + ((a1) * (a1)) + ((a2) * (a2)));
    grad_d[0]  = a0 * a3;
    grad_d[1]  = a1 * a3;
    grad_d[2]  = a2 * a3;
    grad_d[3]  = -a0 * a3;
    grad_d[4]  = -a1 * a3;
    grad_d[5]  = -a2 * a3;
    return grad_d;
}

template <class TScalar>
math::linalg::mini::SVector<TScalar, 9>
SymbolicPointEdgeGradient(math::linalg::mini::SVector<TScalar, 9> const& x)
{
    auto x_0 = x(0);
    auto x_1 = x(1);
    auto x_2 = x(2);
    auto a_0 = x(3);
    auto a_1 = x(4);
    auto a_2 = x(5);
    auto b_0 = x(6);
    auto b_1 = x(7);
    auto b_2 = x(8);
    math::linalg::mini::SVector<TScalar, 9> grad_d;
    TScalar a0  = 2 * a_1;
    TScalar a1  = 2 * b_1;
    TScalar a2  = a0 - a1;
    TScalar a3  = a_0 - b_0;
    TScalar a4  = -a3;
    TScalar a5  = -a_1 + x_1;
    TScalar a6  = a_0 - x_0;
    TScalar a7  = a_1 - b_1;
    TScalar a8  = -a7;
    TScalar a9  = a4 * a5 + a6 * a8;
    TScalar a10 = (1.0 / 2.0) * a9;
    TScalar a11 = a_2 - x_2;
    TScalar a12 = a_2 - b_2;
    TScalar a13 = a11 * a3 - a12 * a6;
    TScalar a14 = 2 * a_2;
    TScalar a15 = 2 * b_2;
    TScalar a16 = (1.0 / 2.0) * a14 - 1.0 / 2.0 * a15;
    TScalar a17 = ((a12) * (a12)) + ((a3) * (a3)) + ((a7) * (a7));
    TScalar a18 = -a12;
    TScalar a19 = -a11 * a8 - a18 * a5;
    TScalar a20 = std::sqrt(((a13) * (a13)) + ((a19) * (a19)) + ((a9) * (a9)));
    TScalar a21 = 1 / (std::sqrt(a17) * a20);
    TScalar a22 = 2 * a_0;
    TScalar a23 = 2 * b_0;
    TScalar a24 = -a22 + a23;
    TScalar a25 = (1.0 / 2.0) * a13;
    TScalar a26 = (1.0 / 2.0) * a19;
    TScalar a27 = a20 / std::pow(a17, 3.0 / 2.0);
    TScalar a28 = -2 * x_1;
    TScalar a29 = a1 + a28;
    TScalar a30 = -2 * x_2;
    TScalar a31 = a15 + a30;
    TScalar a32 = -2 * x_0;
    TScalar a33 = -a23 - a32;
    TScalar a34 = a0 + a28;
    TScalar a35 = -a14 - a30;
    TScalar a36 = a22 + a32;
    grad_d[0]   = a21 * (a10 * a2 + a13 * a16);
    grad_d[1]   = a21 * (a10 * a24 + a16 * a19);
    grad_d[2]   = a21 * (-a2 * a26 + a24 * a25);
    grad_d[3]   = a21 * (a10 * a29 + a25 * a31) + a27 * a4;
    grad_d[4]   = a21 * (a10 * a33 + a26 * a31) + a27 * a8;
    grad_d[5]   = a18 * a27 + a21 * (a25 * a33 - a26 * a29);
    grad_d[6]   = a21 * (-a10 * a34 + a25 * a35) + a27 * a3;
    grad_d[7]   = a21 * (a10 * a36 + a26 * a35) + a27 * a7;
    grad_d[8]   = a12 * a27 + a21 * (a25 * a36 + a26 * a34);
    return grad_d;
}

template <class TScalar>
math::linalg::mini::SVector<TScalar, 12>
SymbolicPointTriangleGradient(math::linalg::mini::SVector<TScalar, 12> const& x)
{
    auto x_0 = x(0);
    auto x_1 = x(1);
    auto x_2 = x(2);
    auto a_0 = x(3);
    auto a_1 = x(4);
    auto a_2 = x(5);
    auto b_0 = x(6);
    auto b_1 = x(7);
    auto b_2 = x(8);
    auto c_0 = x(9);
    auto c_1 = x(10);
    auto c_2 = x(11);
    math::linalg::mini::SVector<TScalar, 12> grad_d;
    TScalar a0  = a_1 - b_1;
    TScalar a1  = -a0;
    TScalar a2  = -c_2;
    TScalar a3  = a2 + a_2;
    TScalar a4  = -a3;
    TScalar a5  = -c_1;
    TScalar a6  = a5 + a_1;
    TScalar a7  = -a6;
    TScalar a8  = a_2 - b_2;
    TScalar a9  = -a8;
    TScalar a10 = a1 * a4 - a7 * a9;
    TScalar a11 = a_0 - b_0;
    TScalar a12 = -a11;
    TScalar a13 = -c_0;
    TScalar a14 = a13 + a_0;
    TScalar a15 = -a14;
    TScalar a16 = -a1 * a15 + a12 * a7;
    TScalar a17 = a11 * a3 - a14 * a8;
    TScalar a18 = ((a10) * (a10)) + ((a16) * (a16)) + ((a17) * (a17));
    TScalar a19 = 1 / std::sqrt(a18);
    TScalar a20 = a10 * a19;
    TScalar a21 = -a12 * a4 + a15 * a9;
    TScalar a22 = a19 * a21;
    TScalar a23 = a16 * a19;
    TScalar a24 = a2 + b_2;
    TScalar a25 = -a_1 + x_1;
    TScalar a26 = a19 * a25;
    TScalar a27 = a5 + b_1;
    TScalar a28 = -a_2 + x_2;
    TScalar a29 = a19 * a28;
    TScalar a30 = std::pow(a18, -3.0 / 2.0);
    TScalar a31 = 2 * b_1;
    TScalar a32 = -2 * c_1;
    TScalar a33 = a31 + a32;
    TScalar a34 = (1.0 / 2.0) * a16;
    TScalar a35 = 2 * b_2;
    TScalar a36 = -2 * c_2;
    TScalar a37 = (1.0 / 2.0) * a35 + (1.0 / 2.0) * a36;
    TScalar a38 = a30 * (-a17 * a37 - a33 * a34);
    TScalar a39 = -a_0 + x_0;
    TScalar a40 = a10 * a39;
    TScalar a41 = a21 * a25;
    TScalar a42 = a16 * a28;
    TScalar a43 = a19 * a39;
    TScalar a44 = a13 + b_0;
    TScalar a45 = 2 * b_0;
    TScalar a46 = -2 * c_0;
    TScalar a47 = -a45 - a46;
    TScalar a48 = a30 * (-a10 * a37 - a34 * a47);
    TScalar a49 = (1.0 / 2.0) * a17;
    TScalar a50 = (1.0 / 2.0) * a10;
    TScalar a51 = a30 * (a33 * a50 - a47 * a49);
    TScalar a52 = 2 * a_1;
    TScalar a53 = a32 + a52;
    TScalar a54 = 2 * a_2;
    TScalar a55 = -a36 - a54;
    TScalar a56 = a30 * (a34 * a53 - a49 * a55);
    TScalar a57 = 2 * a_0;
    TScalar a58 = a46 + a57;
    TScalar a59 = a30 * (-a34 * a58 - a50 * a55);
    TScalar a60 = a30 * (-a49 * a58 - a50 * a53);
    TScalar a61 = -a31 + a52;
    TScalar a62 = -a35 + a54;
    TScalar a63 = a30 * (-a34 * a61 - a49 * a62);
    TScalar a64 = a45 - a57;
    TScalar a65 = a30 * (-a34 * a64 - a50 * a62);
    TScalar a66 = a30 * (-a49 * a64 + a50 * a61);
    grad_d[0]   = a20;
    grad_d[1]   = a22;
    grad_d[2]   = a23;
    grad_d[3]   = -a20 - a24 * a26 + a27 * a29 + a38 * a40 + a38 * a41 + a38 * a42;
    grad_d[4]   = -a22 + a24 * a43 - a29 * a44 + a40 * a48 + a41 * a48 + a42 * a48;
    grad_d[5]   = -a23 + a26 * a44 - a27 * a43 + a40 * a51 + a41 * a51 + a42 * a51;
    grad_d[6]   = a26 * a3 + a29 * a7 + a40 * a56 + a41 * a56 + a42 * a56;
    grad_d[7]   = a14 * a29 + a4 * a43 + a40 * a59 + a41 * a59 + a42 * a59;
    grad_d[8]   = a15 * a26 + a40 * a60 + a41 * a60 + a42 * a60 + a43 * a6;
    grad_d[9]   = a0 * a29 + a26 * a9 + a40 * a63 + a41 * a63 + a42 * a63;
    grad_d[10]  = a12 * a29 + a40 * a65 + a41 * a65 + a42 * a65 + a43 * a8;
    grad_d[11]  = a1 * a43 + a11 * a26 + a40 * a66 + a41 * a66 + a42 * a66;
    return grad_d;
}

template <class TScalar>
math::linalg::mini::SVector<TScalar, 12>
SymbolicEdgeEdgeGradient(math::linalg::mini::SVector<TScalar, 12> const& x, TScalar eps)
{
    auto a_0 = x(0);
    auto a_1 = x(1);
    auto a_2 = x(2);
    auto b_0 = x(3);
    auto b_1 = x(4);
    auto b_2 = x(5);
    auto c_0 = x(6);
    auto c_1 = x(7);
    auto c_2 = x(8);
    auto d_0 = x(9);
    auto d_1 = x(10);
    auto d_2 = x(11);
    math::linalg::mini::SVector<TScalar, 12> grad_d;
    TScalar a0  = c_2 - d_2;
    TScalar a1  = -a0;
    TScalar a2  = a_0 - b_0;
    TScalar a3  = -a2;
    TScalar a4  = c_1 - d_1;
    TScalar a5  = -a4;
    TScalar a6  = a_1 - b_1;
    TScalar a7  = -a6;
    TScalar a8  = c_0 - d_0;
    TScalar a9  = -a8;
    TScalar a10 = a3 * a5 - a7 * a9;
    TScalar a11 = a_2 - b_2;
    TScalar a12 = -a11;
    TScalar a13 = -a1 * a3 + a12 * a9;
    TScalar a14 = a1 * a7 - a12 * a5;
    TScalar a15 = ((a10) * (a10)) + ((a13) * (a13)) + ((a14) * (a14)) + ((eps) * (eps));
    TScalar a16 = 1 / std::sqrt(a15);
    TScalar a17 = -a_1 + c_1;
    TScalar a18 = a16 * a17;
    TScalar a19 = -a_2 + c_2;
    TScalar a20 = a16 * a19;
    TScalar a21 = a14 * a16;
    TScalar a22 = std::pow(a15, -3.0 / 2.0);
    TScalar a23 = 2 * c_1 - 2 * d_1;
    TScalar a24 = (1.0 / 2.0) * a10;
    TScalar a25 = 2 * c_2 - 2 * d_2;
    TScalar a26 = -a25;
    TScalar a27 = (1.0 / 2.0) * a13;
    TScalar a28 = a22 * (-a23 * a24 - a26 * a27);
    TScalar a29 = -a_0 + c_0;
    TScalar a30 = a14 * a29;
    TScalar a31 = a13 * a17;
    TScalar a32 = a10 * a19;
    TScalar a33 = a16 * a29;
    TScalar a34 = a13 * a16;
    TScalar a35 = 2 * c_0 - 2 * d_0;
    TScalar a36 = -a35;
    TScalar a37 = (1.0 / 2.0) * a14;
    TScalar a38 = a22 * (-a24 * a36 - a25 * a37);
    TScalar a39 = a10 * a16;
    TScalar a40 = -a23;
    TScalar a41 = a22 * (-a27 * a35 - a37 * a40);
    TScalar a42 = a22 * (-a24 * a40 - a25 * a27);
    TScalar a43 = a22 * (-a24 * a35 - a26 * a37);
    TScalar a44 = a22 * (-a23 * a37 - a27 * a36);
    TScalar a45 = 2 * a_1 - 2 * b_1;
    TScalar a46 = -a45;
    TScalar a47 = 2 * a_2 - 2 * b_2;
    TScalar a48 = a22 * (-a24 * a46 - a27 * a47);
    TScalar a49 = 2 * a_0 - 2 * b_0;
    TScalar a50 = -a47;
    TScalar a51 = a22 * (-a24 * a49 - a37 * a50);
    TScalar a52 = -a49;
    TScalar a53 = a22 * (-a27 * a52 - a37 * a45);
    TScalar a54 = a22 * (-a24 * a45 - a27 * a50);
    TScalar a55 = a22 * (-a24 * a52 - a37 * a47);
    TScalar a56 = a22 * (-a27 * a49 - a37 * a46);
    grad_d[0]   = a1 * a18 + a20 * a4 - a21 + a28 * a30 + a28 * a31 + a28 * a32;
    grad_d[1]   = a0 * a33 + a20 * a9 + a30 * a38 + a31 * a38 + a32 * a38 - a34;
    grad_d[2]   = a18 * a8 + a30 * a41 + a31 * a41 + a32 * a41 + a33 * a5 - a39;
    grad_d[3]   = a0 * a18 + a20 * a5 + a30 * a42 + a31 * a42 + a32 * a42;
    grad_d[4]   = a1 * a33 + a20 * a8 + a30 * a43 + a31 * a43 + a32 * a43;
    grad_d[5]   = a18 * a9 + a30 * a44 + a31 * a44 + a32 * a44 + a33 * a4;
    grad_d[6]   = a11 * a18 + a20 * a7 + a21 + a30 * a48 + a31 * a48 + a32 * a48;
    grad_d[7]   = a12 * a33 + a2 * a20 + a30 * a51 + a31 * a51 + a32 * a51 + a34;
    grad_d[8]   = a18 * a3 + a30 * a53 + a31 * a53 + a32 * a53 + a33 * a6 + a39;
    grad_d[9]   = a12 * a18 + a20 * a6 + a30 * a54 + a31 * a54 + a32 * a54;
    grad_d[10]  = a11 * a33 + a20 * a3 + a30 * a55 + a31 * a55 + a32 * a55;
    grad_d[11]  = a18 * a2 + a30 * a56 + a31 * a56 + a32 * a56 + a33 * a7;
    return grad_d;
}

} // namespace pbat::geometry::test
