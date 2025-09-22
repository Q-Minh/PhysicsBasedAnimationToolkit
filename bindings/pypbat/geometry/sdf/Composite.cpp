#include "Composite.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <pbat/Aliases.h>
#include <pbat/geometry/sdf/Composite.h>
#include <pbat/geometry/sdf/Forest.h>
#include <pbat/io/Archive.h>
#include <pbat/math/linalg/mini/Eigen.h>
#include <span>
#include <utility>
#include <vector>

namespace pbat::py::geometry::sdf {

void BindComposite(nanobind::module_& m)
{
    namespace nb     = nanobind;
    using ScalarType = Scalar;
    using Composite  = pbat::geometry::sdf::Composite<ScalarType>;
    using Node       = pbat::geometry::sdf::Node<ScalarType>;
    using Transform  = pbat::geometry::sdf::Transform<ScalarType>;
    using Forest     = pbat::geometry::sdf::Forest<ScalarType>;
    using MatX       = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecX       = Eigen::Vector<ScalarType, Eigen::Dynamic>;
    using Vec3       = Eigen::Vector<ScalarType, 3>;
    using pbat::math::linalg::mini::FromEigen;

    nb::class_<Forest>(m, "Forest")
        .def(nb::init<>())
        .def(
            "__init__",
            [](Forest* self,
               std::vector<Node> const& nodes,
               std::vector<Transform> const& transforms,
               std::vector<std::pair<int, int>> const& children,
               std::vector<int> const& roots) {
                new (self) Forest();
                self->nodes      = nodes;
                self->transforms = transforms;
                self->children   = children;
                self->roots      = roots;
            },
            nb::arg("nodes"),
            nb::arg("transforms"),
            nb::arg("children"),
            nb::arg("roots"),
            "Construct a Forest structure from nodes, transforms, children and roots\n\n"
            "Args:\n"
            "    nodes (List[Node]): List of nodes in the forest\n"
            "    transforms (List[Transform]): List of transforms associated to each node\n"
            "    children (List[Tuple[int, int]]): List of pairs of children indices for each "
            "node, such that c* < 0 if no child\n"
            "    roots (List[int]): List of root node indices in the forest")
        .def_rw("nodes", &Forest::nodes, "(List[Node]) List of nodes in the forest")
        .def_rw(
            "transforms",
            &Forest::transforms,
            "(List[Transform]) List of transforms associated to each node")
        .def_rw("roots", &Forest::roots, "(List[int]) List of root node indices in the forest")
        .def_rw(
            "children",
            &Forest::children,
            "(List[Tuple[int, int]]) List of pairs of children indices for each node, such that "
            "c* < 0 if no child")
        .def(
            "serialize",
            &Forest::Serialize,
            nb::arg("archive"),
            "Serialize the forest to an archive\n\n"
            "Args:\n"
            "    archive (Archive): Archive to serialize to")
        .def(
            "deserialize",
            &Forest::Deserialize,
            nb::arg("archive"),
            "Deserialize the forest from an archive\n\n"
            "Args:\n"
            "    archive (Archive): Archive to deserialize from");

    nb::enum_<pbat::geometry::sdf::ECompositeStatus>(m, "ECompositeStatus")
        .value("Valid", pbat::geometry::sdf::ECompositeStatus::Valid)
        .value("InvalidForest", pbat::geometry::sdf::ECompositeStatus::InvalidForest)
        .value("UnexpectedNodeType", pbat::geometry::sdf::ECompositeStatus::UnexpectedNodeType)
        .export_values();

    nb::class_<Composite>(m, "Composite")
        .def(
            "__init__",
            [](Composite* self, Forest const& forest) {
                new (self) Composite(
                    std::span<Node const>(forest.nodes),
                    std::span<Transform const>(forest.transforms),
                    std::span<std::pair<int, int> const>(forest.children),
                    std::span<int const>(forest.roots));
            },
            nb::arg("forest"),
            "Construct a Composite SDF from a Forest\n\n"
            "Args:\n"
            "    forest (Forest): The input forest structure\n")
        .def_prop_ro("status", &Composite::Status, "Get the status of the composite SDF")
        .def(
            "eval",
            [](Composite& self, Vec3 const& p) -> ScalarType { return self.Eval(FromEigen(p)); },
            nb::arg("p"),
            "Evaluate the signed distance function at a given point\n\n"
            "Args:\n"
            "    p (Vec3): The point at which to evaluate the SDF\n"
            "Returns:\n"
            "    float: Signed distance to the composite shape (negative inside, positive "
            "outside)")
        .def(
            "eval",
            [](Composite const& self, MatX const& P) -> VecX {
                VecX sd(P.cols());
                for (int i = 0; i < P.cols(); ++i)
                    sd(i) = self.Eval(FromEigen(P.col(i).head<3>()));
                return sd;
            },
            nb::arg("P"),
            "Evaluate the signed distance function at multiple points\n\n"
            "Args:\n"
            "    P (numpy.ndarray): `3 x N` array of points at which to evaluate the SDF\n"
            "Returns:\n"
            "    numpy.ndarray: `N x 1` array of signed distances to the composite shape "
            "(negative inside, positive outside)")
        .def(
            "grad",
            [](Composite& self, Vec3 const& p, ScalarType h) -> Vec3 {
                return ToEigen(self.Grad(FromEigen(p), h));
            },
            nb::arg("p"),
            nb::arg("h") = ScalarType(1e-4),
            "Evaluate the (numerical) gradient of the signed distance function at a given point\n\n"
            "Args:\n"
            "    p (Vec3): The point at which to evaluate the gradient\n"
            "    h (float): Finite difference step size\n"
            "Returns:\n"
            "    numpy.ndarray: Numerical gradient of the signed distance function at point p")
        .def(
            "grad",
            [](Composite const& self, MatX const& P, ScalarType h) -> MatX {
                MatX G(3, P.cols());
                for (int i = 0; i < P.cols(); ++i)
                    G.col(i) = ToEigen(self.Grad(FromEigen(P.col(i).head<3>()), h));
                return G;
            },
            nb::arg("P"),
            nb::arg("h") = ScalarType(1e-4),
            "Evaluate the (numerical) gradient of the signed distance function at multiple "
            "points\n\n"
            "Args:\n"
            "    P (numpy.ndarray): `3 x N` array of points at which to evaluate the gradient\n"
            "    h (float): Finite difference step size\n"
            "Returns:\n"
            "    numpy.ndarray: `3 x N` array of numerical gradients of the signed distance "
            "function at the input points");

    m.def(
        "roots_and_parents",
        [](std::vector<std::pair<int, int>> const& children)
            -> std::pair<std::vector<int>, std::vector<int>> {
            return pbat::geometry::sdf::FindRootsAndParents(
                std::span<std::pair<int, int> const>(children));
        },
        nb::arg("children"),
        "Find the roots and parents of the composite SDF from the children list\n\n"
        "Args:\n"
        "    children (List[Tuple[int, int]]): List of pairs of children indices for each "
        "node, "
        "such that c* < 0 if no child\n\n"
        "Returns:\n"
        "    Tuple[List[int], List[int]]: Tuple containing:\n"
        "        - List of root indices of the composite SDF\n"
        "        - List of parent node indices, such that parents[n] = -1 if n is a root");
}

} // namespace pbat::py::geometry::sdf