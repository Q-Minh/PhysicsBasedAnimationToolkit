#include "UnaryNode.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <pbat/geometry/sdf/UnaryNode.h>
#include <pbat/math/linalg/mini/Eigen.h>

namespace pbat::py::geometry::sdf {

void BindUnaryNode(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using VecX       = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;
    using MatX       = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vec3       = Eigen::Vector<ScalarType, 3>;
    using SdfVec3    = pbat::geometry::sdf::Vec3<ScalarType>;
    using pbat::math::linalg::mini::FromEigen;
    using pbat::math::linalg::mini::ToEigen;

    using Scale = pbat::geometry::sdf::Scale<ScalarType>;
    nb::class_<Scale>(m, "Scale")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Scale* self, ScalarType s) {
                new (self) Scale();
                self->s = s;
            },
            nb::arg("s"),
            "Constructor with scaling factor\n\n"
            "Args:\n"
            "    s (float): Uniform scaling factor")
        .def_rw("s", &Scale::s, "(float) Uniform scaling factor")
        .def(
            "eval",
            [](Scale const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to scale\n\n"
            "Returns:\n"
            "    float: Signed distance to the scaled shape (negative inside, positive outside)")
        .def(
            "eval",
            [](Scale const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to scale\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the scaled shape (negative inside, "
            "positive outside)");

    using Elongate = pbat::geometry::sdf::Elongate<ScalarType>;
    nb::class_<Elongate>(m, "Elongate")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Elongate* self, Vec3 const& h) {
                new (self) Elongate();
                self->h = FromEigen(h);
            },
            nb::arg("h"),
            "Constructor with elongation\n\n"
            "Args:\n"
            "    h (numpy.ndarray): `3 x 1` elongation along each axis")
        .def_prop_rw(
            "h",
            [](Elongate& self) -> Vec3 { return ToEigen(self.h); },
            [](Elongate& self, Vec3 const& h) { self.h = FromEigen(h); },
            "(numpy.ndarray) `3 x 1` elongation along each axis")
        .def(
            "eval",
            [](Elongate const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to elongate\n\n"
            "Returns:\n"
            "    float: Signed distance to the elongated shape (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](Elongate const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to elongate\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the elongated shape (negative inside, "
            "positive outside)");

    using Round = pbat::geometry::sdf::Round<ScalarType>;
    nb::class_<Round>(m, "Round")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Round* self, ScalarType r) {
                new (self) Round();
                self->r = r;
            },
            nb::arg("r"),
            "Constructor with rounding radius\n\n"
            "Args:\n"
            "    r (float): Rounding radius")
        .def_rw("r", &Round::r, "(float) Rounding radius")
        .def(
            "eval",
            [](Round const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to round\n\n"
            "Returns:\n"
            "    float: Signed distance to the rounded shape (negative inside, positive outside)")
        .def(
            "eval",
            [](Round const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to round\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the rounded shape (negative inside, "
            "positive outside)");

    using Onion = pbat::geometry::sdf::Onion<ScalarType>;
    nb::class_<Onion>(m, "Onion")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Onion* self, ScalarType t) {
                new (self) Onion();
                self->t = t;
            },
            nb::arg("t"),
            "Constructor with onion thickness\n\n"
            "Args:\n"
            "    t (float): Onion thickness")
        .def_rw("t", &Onion::t, "(float) Onion thickness")
        .def(
            "eval",
            [](Onion const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to onion\n\n"
            "Returns:\n"
            "    float: Signed distance to the onion shape (negative inside, positive outside)")

        .def(
            "eval",
            [](Onion const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to onion\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the onion shape (negative inside, "
            "positive outside)");

    using Symmetrize = pbat::geometry::sdf::Symmetrize<ScalarType>;
    nb::class_<Symmetrize>(m, "Symmetrize")
        .def(nb::init<>())
        .def(
            "eval",
            [](Symmetrize const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to symmetrize\n\n"
            "Returns:\n"
            "    float: Signed distance to the symmetrized shape (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](Symmetrize const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to symmetrize\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the symmetrized shape (negative "
            "inside, positive outside)");

    using Repeat = pbat::geometry::sdf::Repeat<ScalarType>;
    nb::class_<Repeat>(m, "Repeat")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Repeat* self, ScalarType s, Vec3 const& l) {
                new (self) Repeat();
                self->s = s;
                self->l = FromEigen(l);
            },
            nb::arg("s"),
            nb::arg("l"),
            "Constructor with repetition cell size\n\n"
            "Args:\n"
            "    s (float): Uniform scaling factor for the repetition\n"
            "    l (numpy.ndarray): `3 x 1` repetition cell size along each axis")
        .def_rw("s", &Repeat::s, "(float) Uniform scaling factor for the repetition")
        .def_prop_rw(
            "l",
            [](Repeat& self) -> Vec3 { return ToEigen(self.l); },
            [](Repeat& self, Vec3 const& l) { self.l = FromEigen(l); },
            "(numpy.ndarray) `3 x 1` repetition cell size along each axis")
        .def(
            "eval",
            [](Repeat const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to repeat\n\n"
            "Returns:\n"
            "    float: Signed distance to the repeated shape (negative inside, positive outside)")
        .def(
            "eval",
            [](Repeat const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to repeat\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the repeated shape (negative inside, "
            "positive outside)");

    using Bump = pbat::geometry::sdf::Bump<ScalarType>;
    nb::class_<Bump>(m, "Bump")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Bump* self, Vec3 const& f, Vec3 const& g) {
                new (self) Bump();
                self->f = FromEigen(f);
                self->g = FromEigen(g);
            },
            nb::arg("f"),
            nb::arg("g"),
            "Constructor with bump radius\n\n"
            "Args:\n"
            "    f (numpy.ndarray): `3 x 1` frequency along each axis"
            "    g (numpy.ndarray): `3 x 1` phase along each axis")
        .def_prop_rw(
            "f",
            [](Bump& self) -> Vec3 { return ToEigen(self.f); },
            [](Bump& self, Vec3 const& f) { self.f = FromEigen(f); },
            "(numpy.ndarray) `3 x 1` frequency along each axis")
        .def_prop_rw(
            "g",
            [](Bump& self) -> Vec3 { return ToEigen(self.g); },
            [](Bump& self, Vec3 const& g) { self.g = FromEigen(g); },
            "(numpy.ndarray) `3 x 1` phase along each axis")
        .def(
            "eval",
            [](Bump const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to bump\n\n"
            "Returns:\n"
            "    float: Signed distance to the bumped shape (negative inside, positive outside)")
        .def(
            "eval",
            [](Bump const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to bump\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the bumped shape (negative inside, "
            "positive outside)");

    using Twist = pbat::geometry::sdf::Twist<ScalarType>;
    nb::class_<Twist>(m, "Twist")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Twist* self, ScalarType k) {
                new (self) Twist();
                self->k = k;
            },
            nb::arg("k"),
            "Constructor with twist factor\n\n"
            "Args:\n"
            "    k (float): Twist factor")
        .def_rw("k", &Twist::k, "(float) Twist factor")
        .def(
            "eval",
            [](Twist const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to twist\n\n"
            "Returns:\n"
            "    float: Signed distance to the twisted shape (negative inside, positive outside)")
        .def(
            "eval",
            [](Twist const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to twist\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the twisted shape (negative inside, "
            "positive outside)");

    using Bend = pbat::geometry::sdf::Bend<ScalarType>;
    nb::class_<Bend>(m, "Bend")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Bend* self, ScalarType k) {
                new (self) Bend();
                self->k = k;
            },
            nb::arg("k"),
            "Constructor with bend factor\n\n"
            "Args:\n"
            "    k (float): Bend factor")
        .def_rw("k", &Bend::k, "(float) Bend factor")
        .def(
            "eval",
            [](Bend const& self,
               Vec3 const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> ScalarType {
                return self.Eval(FromEigen(p), [&](SdfVec3 const& x) -> ScalarType {
                    return sdf(ToEigen(x));
                });
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at a point\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `3 x 1` point in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to bend\n\n"
            "Returns:\n"
            "    float: Signed distance to the bent shape (negative inside, positive outside)")
        .def(
            "eval",
            [](Bend const& self,
               MatX const& p,
               std::function<ScalarType(Vec3 const&)> const& sdf) -> VecX {
                VecX result(p.cols());
                for (int i = 0; i < p.cols(); ++i)
                {
                    result(i) = self.Eval(
                        FromEigen(p.col(i).head<3>()),
                        [&](SdfVec3 const& x) -> ScalarType { return sdf(ToEigen(x)); });
                }
                return result;
            },
            nb::arg("p"),
            nb::arg("sdf"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    p (numpy.ndarray): `N x 3` points in 3D space\n"
            "    sdf (Callable): Callable with signature `float(numpy.ndarray)` representing the "
            "SDF to bend\n\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` Signed distances to the bent shape (negative inside, "
            "positive outside)");
}

} // namespace pbat::py::geometry::sdf