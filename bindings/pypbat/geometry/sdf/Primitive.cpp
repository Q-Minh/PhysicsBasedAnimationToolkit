#include "Primitive.h"

#include <nanobind/eigen/dense.h>
#include <pbat/geometry/sdf/Primitive.h>
#include <pbat/math/linalg/mini/Eigen.h>

namespace pbat::py::geometry::sdf {

void BindPrimitive(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using VecX       = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
    using MatX       = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
    using Mat3       = Eigen::Matrix<ScalarType, 3, 3>;
    using Vec3       = Eigen::Vector<ScalarType, 3>;
    using Vec2       = Eigen::Vector<ScalarType, 2>;
    using pbat::math::linalg::mini::FromEigen;
    using pbat::math::linalg::mini::ToEigen;
    using Sphere = pbat::geometry::sdf::Sphere<ScalarType>;
    nb::class_<Sphere>(m, "Sphere")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Sphere* self, ScalarType R) {
                new (self) Sphere();
                self->R = R;
            },
            nb::arg("R"),
            "Constructor with radius\n\n"
            "Args:\n"
            "    R (float): Sphere radius")
        .def_rw("R", &Sphere::R, "(float) Sphere radius")
        .def(
            "eval",
            [](Sphere const& self, Vec3 const& p) -> ScalarType { return self.Eval(FromEigen(p)); },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the sphere (negative inside, positive outside)")
        .def(
            "eval",
            [](Sphere const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the sphere (negative inside, positive "
            "outside)");

    using Box = pbat::geometry::sdf::Box<ScalarType>;
    nb::class_<Box>(m, "Box")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Box* self, Vec3 const& he) {
                new (self) Box();
                self->he = FromEigen(he);
            },
            nb::arg("he"),
            "Constructor with half extents\n\n"
            "Args:\n"
            "    he (numpy.ndarray): `3 x 1` half extents of the box along each axis")
        .def_prop_rw(
            "he",
            [](Box& self) -> Vec3 { return ToEigen(self.he); },
            [](Box& self, Vec3 const& he) { self.he = FromEigen(he); },
            "(numpy.ndarray) `3 x 1` half extents of the box along each axis")
        .def(
            "eval",
            [](Box const& self, Vec3 const& p) -> ScalarType { return self.Eval(FromEigen(p)); },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the box (negative inside, positive outside)")
        .def(
            "eval",
            [](Box const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the box (negative inside, positive "
            "outside)");

    using BoxFrame = pbat::geometry::sdf::BoxFrame<ScalarType>;
    nb::class_<BoxFrame>(m, "BoxFrame")
        .def(nb::init<>())
        .def(
            "__init__",
            [](BoxFrame* self, Vec3 const& he, ScalarType t) {
                new (self) BoxFrame();
                self->he = FromEigen(he);
                self->t  = t;
            },
            nb::arg("he"),
            nb::arg("t"),
            "Constructor with half extents and thickness\n\n"
            "Args:\n"
            "    he (numpy.ndarray): `3 x 1` half extents of the box frame along each axis\n"
            "    t (float): Thickness of the box frame")
        .def_prop_rw(
            "he",
            [](BoxFrame& self) -> Vec3 { return ToEigen(self.he); },
            [](BoxFrame& self, Vec3 const& he) { self.he = FromEigen(he); },
            "(numpy.ndarray) `3 x 1` half extents of the box frame along each axis")
        .def_rw("t", &BoxFrame::t, "(float) Thickness of the box frame")
        .def(
            "eval",
            [](BoxFrame const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the box frame (negative inside, positive outside)")
        .def(
            "eval",
            [](BoxFrame const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the box frame (negative inside, "
            "positive outside)");

    using Torus = pbat::geometry::sdf::Torus<ScalarType>;
    nb::class_<Torus>(m, "Torus")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Torus* self, Vec2 const& t) {
                new (self) Torus();
                self->t = FromEigen(t);
            },
            nb::arg("t"),
            "Constructor with minor and major radius\n\n"
            "Args:\n"
            "    t (numpy.ndarray): `2 x 1` minor and major radius of the torus")
        .def_prop_rw(
            "t",
            [](Torus& self) -> Eigen::Matrix<ScalarType, 2, 1> { return ToEigen(self.t); },
            [](Torus& self, Eigen::Matrix<ScalarType, 2, 1> const& t) { self.t = FromEigen(t); },
            "(numpy.ndarray) `2 x 1` minor and major radius of the torus")
        .def(
            "eval",
            [](Torus const& self, Vec3 const& p) -> ScalarType { return self.Eval(FromEigen(p)); },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the torus (negative inside, positive outside)")
        .def(
            "eval",
            [](Torus const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the Torus (negative inside, "
            "positive outside)");

    using CappedTorus = pbat::geometry::sdf::CappedTorus<ScalarType>;
    nb::class_<CappedTorus>(m, "CappedTorus")
        .def(nb::init<>())
        .def(
            "__init__",
            [](CappedTorus* self, Vec2 const& sc, ScalarType ra, ScalarType rb) {
                new (self) CappedTorus();
                self->sc = FromEigen(sc);
                self->ra = ra;
                self->rb = rb;
            },
            nb::arg("sc"),
            nb::arg("ra"),
            nb::arg("rb"),
            "Constructor with minor and major radius and cap height\n\n"
            "Args:\n"
            "    sc (numpy.ndarray): `2 x 1` minor and major radius of the capped torus\n"
            "    ra (float): Unknown\n"
            "    rb (float): Unknown\n")
        .def_prop_rw(
            "sc",
            [](CappedTorus& self) -> Vec2 { return ToEigen(self.sc); },
            [](CappedTorus& self, Vec2 const& sc) { self.sc = FromEigen(sc); },
            "(numpy.ndarray) `2 x 1` minor and major radius of the capped torus")
        .def_rw("ra", &CappedTorus::ra, "(float) Unknown")
        .def_rw("rb", &CappedTorus::rb, "(float) Unknown")
        .def(
            "eval",
            [](CappedTorus const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the capped torus (negative inside, positive outside)")
        .def(
            "eval",
            [](CappedTorus const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the capped torus (negative inside, "
            "positive outside)");

    using Link = pbat::geometry::sdf::Link<ScalarType>;
    nb::class_<Link>(m, "Link")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Link* self, Vec2 const& t, ScalarType le) {
                new (self) Link();
                self->t  = FromEigen(t);
                self->le = le;
            },
            nb::arg("t"),
            nb::arg("le"),
            "Constructor with rotation matrix, position vector, and radius\n\n"
            "Args:\n"
            "    t (numpy.ndarray): `2 x 1` t[0]: minor radius, t[1]: major radius of the link\n"
            "    le (float): Elongation length of the link")
        .def_prop_rw(
            "t",
            [](Link& self) -> Vec2 { return ToEigen(self.t); },
            [](Link& self, Vec2 const& t) { self.t = FromEigen(t); },
            "(numpy.ndarray) `2 x 1` t[0]: minor radius, t[1]: major radius of the link")
        .def_rw("le", &Link::le, "(float) Elongation length of the link")
        .def(
            "eval",
            [](Link const& self, Vec3 const& x) -> ScalarType { return self.Eval(FromEigen(x)); },
            nb::arg("x"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    x (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the link (negative inside, positive outside)")
        .def(
            "eval",
            [](Link const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the link (negative inside, "
            "positive outside)");

    using InfiniteCylinder = pbat::geometry::sdf::InfiniteCylinder<ScalarType>;
    nb::class_<InfiniteCylinder>(m, "InfiniteCylinder")
        .def(nb::init<>())
        .def(
            "__init__",
            [](InfiniteCylinder* self, Vec3 const& c) {
                new (self) InfiniteCylinder();
                self->c = FromEigen(c);
            },
            nb::arg("c"),
            "Constructor with center and radius\n\n"
            "Args:\n"
            "    c (numpy.ndarray): `3 x 1` center of the cylinder (on the axis) in c(0), c(1) and "
            "radius in c(2)")
        .def_prop_rw(
            "c",
            [](InfiniteCylinder& self) -> Vec3 { return ToEigen(self.c); },
            [](InfiniteCylinder& self, Vec3 const& c) { self.c = FromEigen(c); },
            "(numpy.ndarray) `3 x 1` center of the cylinder (on the axis) in c(0), c(1) and radius "
            "in c(2)")
        .def(
            "eval",
            [](InfiniteCylinder const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the infinite cylinder (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](InfiniteCylinder const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the InfiniteCylinder (negative inside, "
            "positive outside)");

    using Cone = pbat::geometry::sdf::Cone<ScalarType>;
    nb::class_<Cone>(m, "Cone")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Cone* self, Vec2 const& c, ScalarType h) {
                new (self) Cone();
                self->c = FromEigen(c);
                self->h = h;
            },
            nb::arg("c"),
            nb::arg("h"),
            "Constructor with sin/cos of the angle and height\n\n"
            "Args:\n"
            "    c (numpy.ndarray): `2 x 1` sin/cos of the angle of the cone\n"
            "    h (float): Height of the cone")
        .def_prop_rw(
            "c",
            [](Cone& self) -> Vec2 { return ToEigen(self.c); },
            [](Cone& self, Vec2 const& c) { self.c = FromEigen(c); },
            "(numpy.ndarray) `2 x 1` sin/cos of the angle of the cone")
        .def_rw("h", &Cone::h, "(float) Height of the cone")
        .def(
            "eval",
            [](Cone const& self, Vec3 const& p) -> ScalarType { return self.Eval(FromEigen(p)); },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the cone (negative inside, positive outside)")
        .def(
            "eval",
            [](Cone const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the Cone (negative inside, "
            "positive outside)");

    using InfiniteCone = pbat::geometry::sdf::InfiniteCone<ScalarType>;
    nb::class_<InfiniteCone>(m, "InfiniteCone")
        .def(nb::init<>())
        .def(
            "__init__",
            [](InfiniteCone* self, Vec2 const& c) {
                new (self) InfiniteCone();
                self->c = FromEigen(c);
            },
            nb::arg("c"),
            "Constructor with sin/cos of the angle\n\n"
            "Args:\n"
            "    c (numpy.ndarray): `2 x 1` sin/cos of the angle of the infinite cone")
        .def_prop_rw(
            "c",
            [](InfiniteCone& self) -> Vec2 { return ToEigen(self.c); },
            [](InfiniteCone& self, Vec2 const& c) { self.c = FromEigen(c); },
            "(numpy.ndarray) `2 x 1` sin/cos of the angle of the infinite cone")
        .def(
            "eval",
            [](InfiniteCone const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the infinite cone (negative inside, positive outside)")
        .def(
            "eval",
            [](InfiniteCone const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the InfiniteCone (negative inside, "
            "positive outside)");

    using Plane = pbat::geometry::sdf::Plane<ScalarType>;
    nb::class_<Plane>(m, "Plane")
        .def(nb::init<>())
        .def(
            "eval",
            [](Plane const& self, Vec3 const& p) -> ScalarType { return self.Eval(FromEigen(p)); },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the plane (negative inside, positive outside)")
        .def(
            "eval",
            [](Plane const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the Plane (negative inside, "
            "positive outside)");

    using HexagonalPrism = pbat::geometry::sdf::HexagonalPrism<ScalarType>;
    nb::class_<HexagonalPrism>(m, "HexagonalPrism")
        .def(nb::init<>())
        .def(
            "__init__",
            [](HexagonalPrism* self, Vec2 const& h) {
                new (self) HexagonalPrism();
                self->h = FromEigen(h);
            },
            nb::arg("h"),
            "Constructor with hexagon dimensions and length\n\n"
            "Args:\n"
            "    h (numpy.ndarray): `2 x 1` h[0]: inradius, h[1]: circumradius of the hexagonal "
            "prism\n")
        .def_prop_rw(
            "h",
            [](HexagonalPrism& self) -> Vec2 { return ToEigen(self.h); },
            [](HexagonalPrism& self, Vec2 const& h) { self.h = FromEigen(h); },
            "(numpy.ndarray) `2 x 1` h[0]: inradius, h[1]: circumradius of the hexagonal prism")
        .def(
            "eval",
            [](HexagonalPrism const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the hexagonal prism (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](HexagonalPrism const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the HexagonalPrism (negative inside, "
            "positive outside)");

    using Capsule = pbat::geometry::sdf::Capsule<ScalarType>;
    nb::class_<Capsule>(m, "Capsule")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Capsule* self, Vec3 const& a, Vec3 const& b, ScalarType r) {
                new (self) Capsule();
                self->a = FromEigen(a);
                self->b = FromEigen(b);
                self->r = r;
            },
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("r"),
            "Constructor with endpoints and radius\n\n"
            "Args:\n"
            "    a (numpy.ndarray): `3 x 1` first endpoint of the capsule\n"
            "    b (numpy.ndarray): `3 x 1` second endpoint of the capsule\n"
            "    r (float): Radius of the capsule")
        .def_prop_rw(
            "a",
            [](Capsule& self) -> Vec3 { return ToEigen(self.a); },
            [](Capsule& self, Vec3 const& a) { self.a = FromEigen(a); },
            "(numpy.ndarray) `3 x 1` first endpoint of the capsule")
        .def_prop_rw(
            "b",
            [](Capsule& self) -> Vec3 { return ToEigen(self.b); },
            [](Capsule& self, Vec3 const& b) { self.b = FromEigen(b); },
            "(numpy.ndarray) `3 x 1` second endpoint of the capsule")
        .def_rw("r", &Capsule::r, "(float) Radius of the capsule")
        .def(
            "eval",
            [](Capsule const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the capsule (negative inside, positive outside)")
        .def(
            "eval",
            [](Capsule const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the Capsule (negative inside, "
            "positive outside)");

    using VerticalCapsule = pbat::geometry::sdf::VerticalCapsule<ScalarType>;
    nb::class_<VerticalCapsule>(m, "VerticalCapsule")
        .def(nb::init<>())
        .def(
            "__init__",
            [](VerticalCapsule* self, ScalarType h, ScalarType r) {
                new (self) VerticalCapsule();
                self->h = h;
                self->r = r;
            },
            nb::arg("h"),
            nb::arg("r"),
            "Constructor with height and radius\n\n"
            "Args:\n"
            "    h (float): Height of the vertical capsule\n"
            "    r (float): Radius of the vertical capsule")
        .def_rw("h", &VerticalCapsule::h, "(float) Height of the vertical capsule")
        .def_rw("r", &VerticalCapsule::r, "(float) Radius of the vertical capsule")
        .def(
            "eval",
            [](VerticalCapsule const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the vertical capsule (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](VerticalCapsule const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the VerticalCapsule (negative inside, "
            "positive outside)");

    using CappedCylinder = pbat::geometry::sdf::CappedCylinder<ScalarType>;
    nb::class_<CappedCylinder>(m, "CappedCylinder")
        .def(nb::init<>())
        .def(
            "__init__",
            [](CappedCylinder* self, Vec3 const& a, Vec3 const& b, ScalarType r) {
                new (self) CappedCylinder();
                self->a = FromEigen(a);
                self->b = FromEigen(b);
                self->r = r;
            },
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("r"),
            "Constructor with height and radius\n\n"
            "Args:\n"
            "    a (numpy.ndarray): `3 x 1` first endpoint of the capped cylinder\n"
            "    b (numpy.ndarray): `3 x 1` second endpoint of the capped cylinder\n"
            "    r (float): Radius of the capped cylinder")
        .def_prop_rw(
            "a",
            [](CappedCylinder& self) -> Vec3 { return ToEigen(self.a); },
            [](CappedCylinder& self, Vec3 const& a) { self.a = FromEigen(a); },
            "(numpy.ndarray) `3 x 1` first endpoint of the capped cylinder")
        .def_prop_rw(
            "b",
            [](CappedCylinder& self) -> Vec3 { return ToEigen(self.b); },
            [](CappedCylinder& self, Vec3 const& b) { self.b = FromEigen(b); },
            "(numpy.ndarray) `3 x 1` second endpoint of the capped cylinder")
        .def_rw("r", &CappedCylinder::r, "(float) Radius of the capped cylinder")
        .def(
            "eval",
            [](CappedCylinder const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the capped cylinder (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](CappedCylinder const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the CappedCylinder (negative inside, "
            "positive outside)");

    using VerticalCappedCylinder = pbat::geometry::sdf::VerticalCappedCylinder<ScalarType>;
    nb::class_<VerticalCappedCylinder>(m, "VerticalCappedCylinder")
        .def(nb::init<>())
        .def(
            "__init__",
            [](VerticalCappedCylinder* self, ScalarType h, ScalarType r) {
                new (self) VerticalCappedCylinder();
                self->h = h;
                self->r = r;
            },
            nb::arg("h"),
            nb::arg("r"),
            "Constructor with height and radius\n\n"
            "Args:\n"
            "    h (float): Height of the vertical capped cylinder\n"
            "    r (float): Radius of the vertical capped cylinder")
        .def_rw("h", &VerticalCappedCylinder::h, "(float) Height of the vertical capped cylinder")
        .def_rw("r", &VerticalCappedCylinder::r, "(float) Radius of the vertical capped cylinder")
        .def(
            "eval",
            [](VerticalCappedCylinder const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the vertical capped cylinder (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](VerticalCappedCylinder const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the VerticalCappedCylinder (negative "
            "inside, "
            "positive outside)");

    using RoundedCylinder = pbat::geometry::sdf::RoundedCylinder<ScalarType>;
    nb::class_<RoundedCylinder>(m, "RoundedCylinder")
        .def(nb::init<>())
        .def(
            "__init__",
            [](RoundedCylinder* self, ScalarType h, ScalarType ra, ScalarType rb) {
                new (self) RoundedCylinder();
                self->h  = h;
                self->ra = ra;
                self->rb = rb;
            },
            nb::arg("h"),
            nb::arg("ra"),
            nb::arg("rb"),
            "Constructor with height, radius, and corner radius\n\n"
            "Args:\n"
            "    h (float): Height of the rounded cylinder\n"
            "    ra (float): Radius of the rounded cylinder\n"
            "    rb (float): Corner radius of the rounded cylinder")
        .def_rw("h", &RoundedCylinder::h, "(float) Height of the rounded cylinder")
        .def_rw("ra", &RoundedCylinder::ra, "(float) Radius of the rounded cylinder")
        .def_rw("rb", &RoundedCylinder::rb, "(float) Corner radius of the rounded cylinder")
        .def(
            "eval",
            [](RoundedCylinder const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the rounded cylinder (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](RoundedCylinder const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the RoundedCylinder (negative "
            "inside, positive outside)");

    using VerticalCappedCone = pbat::geometry::sdf::VerticalCappedCone<ScalarType>;
    nb::class_<VerticalCappedCone>(m, "VerticalCappedCone")
        .def(nb::init<>())
        .def(
            "__init__",
            [](VerticalCappedCone* self, ScalarType h, ScalarType r1, ScalarType r2) {
                new (self) VerticalCappedCone();
                self->h  = h;
                self->r1 = r1;
                self->r2 = r2;
            },
            nb::arg("h"),
            nb::arg("r1"),
            nb::arg("r2"),
            "Constructor with height and two radii\n\n"
            "Args:\n"
            "    h (float): Height of the vertical capped cone\n"
            "    r1 (float): Radius of one end of the vertical capped cone\n"
            "    r2 (float): Radius of the other end of the vertical capped cone")
        .def_rw("h", &VerticalCappedCone::h, "(float) Height of the vertical capped cone")
        .def_rw(
            "r1",
            &VerticalCappedCone::r1,
            "(float) Radius of one end of the vertical capped cone")
        .def_rw(
            "r2",
            &VerticalCappedCone::r2,
            "(float) Radius of the other end of the vertical capped cone")
        .def(
            "eval",
            [](VerticalCappedCone const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the vertical capped cone (negative inside, positive "
            "outside")
        .def(
            "eval",
            [](VerticalCappedCone const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the VerticalCappedCone (negative "
            "inside, positive outside)");

    using CutHollowSphere = pbat::geometry::sdf::CutHollowSphere<ScalarType>;
    nb::class_<CutHollowSphere>(m, "CutHollowSphere")
        .def(nb::init<>())
        .def(
            "__init__",
            [](CutHollowSphere* self, ScalarType r, ScalarType h, ScalarType t) {
                new (self) CutHollowSphere();
                self->r = r;
                self->h = h;
                self->t = t;
            },
            nb::arg("r"),
            nb::arg("h"),
            nb::arg("t"),
            "Constructor with radius and thickness\n\n"
            "Args:\n"
            "    r (float): Radius of the cut hollow sphere\n"
            "    h (float): Height of the cut hollow sphere\n"
            "    t (float): Thickness of the cut hollow sphere")
        .def_rw("r", &CutHollowSphere::r, "(float) Radius of the cut hollow sphere")
        .def_rw("h", &CutHollowSphere::h, "(float) Height of the cut hollow sphere")
        .def_rw("t", &CutHollowSphere::t, "(float) Thickness of the cut hollow sphere")
        .def(
            "eval",
            [](CutHollowSphere const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the cut hollow sphere (negative inside, positive "
            "outside")
        .def(
            "eval",
            [](CutHollowSphere const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the CutHollowSphere (negative "
            "inside, positive outside)");

    using VerticalRoundCone = pbat::geometry::sdf::VerticalRoundCone<ScalarType>;
    nb::class_<VerticalRoundCone>(m, "VerticalRoundCone")
        .def(nb::init<>())
        .def(
            "__init__",
            [](VerticalRoundCone* self, ScalarType h, ScalarType r1, ScalarType r2) {
                new (self) VerticalRoundCone();
                self->h  = h;
                self->r1 = r1;
                self->r2 = r2;
            },
            nb::arg("h"),
            nb::arg("r1"),
            nb::arg("r2"),
            "Constructor with height, two radii, and corner radius\n\n"
            "Args:\n"
            "    h (float): Height of the vertical round cone\n"
            "    r1 (float): Radius of one end of the vertical round cone\n"
            "    r2 (float): Radius of the other end of the vertical round cone\n")
        .def_rw("h", &VerticalRoundCone::h, "(float) Height of the vertical round cone")
        .def_rw(
            "r1",
            &VerticalRoundCone::r1,
            "(float) Radius of one end of the vertical round cone")
        .def_rw(
            "r2",
            &VerticalRoundCone::r2,
            "(float) Radius of the other end of the vertical round cone")
        .def(
            "eval",
            [](VerticalRoundCone const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the vertical round cone (negative inside, positive "
            "outside")
        .def(
            "eval",
            [](VerticalRoundCone const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the VerticalRoundCone (negative "
            "inside, positive outside)");

    using Octahedron = pbat::geometry::sdf::Octahedron<ScalarType>;
    nb::class_<Octahedron>(m, "Octahedron")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Octahedron* self, ScalarType s) {
                new (self) Octahedron();
                self->s = s;
            },
            nb::arg("s"),
            "Constructor with extent\n\n"
            "Args:\n"
            "    s (float): Extent of the octahedron from center to a vertex")
        .def_rw("s", &Octahedron::s, "(float) Extent of the octahedron from center to a vertex")
        .def(
            "eval",
            [](Octahedron const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the octahedron (negative inside, positive outside)")
        .def(
            "eval",
            [](Octahedron const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the Octahedron (negative "
            "inside, positive outside)");

    using Pyramid = pbat::geometry::sdf::Pyramid<ScalarType>;
    nb::class_<Pyramid>(m, "Pyramid")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Pyramid* self, ScalarType h) {
                new (self) Pyramid();
                self->h = h;
            },
            nb::arg("h"),
            "Constructor with base half extents and height\n\n"
            "Args:\n"
            "    h (float): Height of the pyramid")
        .def_rw("h", &Pyramid::h, "(float) Height of the pyramid")
        .def(
            "eval",
            [](Pyramid const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the pyramid (negative inside, positive outside)")
        .def(
            "eval",
            [](Pyramid const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the Pyramid (negative "
            "inside, positive outside)");

    using Triangle = pbat::geometry::sdf::Triangle<ScalarType>;
    nb::class_<Triangle>(m, "Triangle")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Triangle* self, Vec3 const& a, Vec3 const& b, Vec3 const& c) {
                new (self) Triangle();
                self->a = FromEigen(a);
                self->b = FromEigen(b);
                self->c = FromEigen(c);
            },
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("c"),
            "Constructor with three vertices\n\n"
            "Args:\n"
            "    a (numpy.ndarray): `3 x 1` first vertex of the triangle\n"
            "    b (numpy.ndarray): `3 x 1` second vertex of the triangle\n"
            "    c (numpy.ndarray): `3 x 1` third vertex of the triangle")
        .def_prop_rw(
            "a",
            [](Triangle& self) -> Vec3 { return ToEigen(self.a); },
            [](Triangle& self, Vec3 const& a) { self.a = FromEigen(a); },
            "(numpy.ndarray) `3 x 1` first vertex of the triangle")
        .def_prop_rw(
            "b",
            [](Triangle& self) -> Vec3 { return ToEigen(self.b); },
            [](Triangle& self, Vec3 const& b) { self.b = FromEigen(b); },
            "(numpy.ndarray) `3 x 1` second vertex of the triangle")
        .def_prop_rw(
            "c",
            [](Triangle& self) -> Vec3 { return ToEigen(self.c); },
            [](Triangle& self, Vec3 const& c) { self.c = FromEigen(c); },
            "(numpy.ndarray) `3 x 1` third vertex of the triangle")
        .def(
            "eval",
            [](Triangle const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the triangle (negative inside, positive outside)")
        .def(
            "eval",
            [](Triangle const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the Triangle (negative "
            "inside, positive outside)");

    using Quadrilateral = pbat::geometry::sdf::Quadrilateral<ScalarType>;
    nb::class_<Quadrilateral>(m, "Quadrilateral")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Quadrilateral* self, Vec3 const& a, Vec3 const& b, Vec3 const& c, Vec3 const& d) {
                new (self) Quadrilateral();
                self->a = FromEigen(a);
                self->b = FromEigen(b);
                self->c = FromEigen(c);
                self->d = FromEigen(d);
            },
            nb::arg("a"),
            nb::arg("b"),
            nb::arg("c"),
            nb::arg("d"),
            "Constructor with four vertices\n\n"
            "Args:\n"
            "    a (numpy.ndarray): `3 x 1` first vertex of the quadrilateral\n"
            "    b (numpy.ndarray): `3 x 1` second vertex of the quadrilateral\n"
            "    c (numpy.ndarray): `3 x 1` third vertex of the quadrilateral\n"
            "    d (numpy.ndarray): `3 x 1` fourth vertex of the quadrilateral")
        .def_prop_rw(
            "a",
            [](Quadrilateral& self) -> Vec3 { return ToEigen(self.a); },
            [](Quadrilateral& self, Vec3 const& a) { self.a = FromEigen(a); },
            "(numpy.ndarray) `3 x 1` first vertex of the quadrilateral")
        .def_prop_rw(
            "b",
            [](Quadrilateral& self) -> Vec3 { return ToEigen(self.b); },
            [](Quadrilateral& self, Vec3 const& b) { self.b = FromEigen(b); },
            "(numpy.ndarray) `3 x 1` second vertex of the quadrilateral")
        .def_prop_rw(
            "c",
            [](Quadrilateral& self) -> Vec3 { return ToEigen(self.c); },
            [](Quadrilateral& self, Vec3 const& c) { self.c = FromEigen(c); },
            "(numpy.ndarray) `3 x 1` third vertex of the quadrilateral")
        .def_prop_rw(
            "d",
            [](Quadrilateral& self) -> Vec3 { return ToEigen(self.d); },
            [](Quadrilateral& self, Vec3 const& d) { self.d = FromEigen(d); },
            "(numpy.ndarray) `3 x 1` fourth vertex of the quadrilateral")
        .def(
            "eval",
            [](Quadrilateral const& self, Vec3 const& p) -> ScalarType {
                return self.Eval(FromEigen(p));
            },
            nb::arg("p"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n\n"
            "Returns:\n"
            "    float: Signed distance to the quadrilateral (negative inside, positive outside)")
        .def(
            "eval",
            [](Quadrilateral const& self, nb::DRef<MatX const> p) -> VecX {
                VecX result(p.cols());
                for (auto i = 0; i < p.cols(); ++i)
                    result(i) = self.Eval(FromEigen(p.col(i).head<3>()));
                return result;
            },
            nb::arg("p"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x N` points in 3D space\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the Quadrilateral (negative "
            "inside, positive outside)");
}

} // namespace pbat::py::geometry::sdf