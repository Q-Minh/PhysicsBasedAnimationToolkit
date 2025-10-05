#include "Gradient.h"

#include "Mesh.h"

#include <pbat/fem/Gradient.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

namespace pbat {
namespace py {
namespace fem {

void BindGradient([[maybe_unused]] nanobind::module_& m)
{
    namespace nb = nanobind;

    using TScalar = pbat::Scalar;
    using TIndex  = pbat::Index;

    m.def(
        "gradient_matrix",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           TIndex nNodes,
           nb::DRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> GNeg,
           EElement eElement,
           int order,
           int dims) {
            Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> G;
            ApplyToElementInDims(eElement, order, dims, [&]<class ElementType, int Dims>() {
                G = pbat::fem::GradientMatrix<ElementType, Dims, Eigen::RowMajor>(
                    E.template topRows<ElementType::kNodes>(),
                    nNodes,
                    eg,
                    GNeg.template topRows<ElementType::kNodes>());
            });
            return G;
        },
        nb::arg("E"),
        nb::arg("n_nodes"),
        nb::arg("eg"),
        nb::arg("GNeg"),
        nb::arg("element"),
        nb::arg("order") = 1,
        nb::arg("dims")  = 3,
        "Construct the gradient operator's sparse matrix representation.\n\n"
        "Args:\n"
        "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
        "elements.\n"
        "    n_nodes (int): Number of mesh nodes.\n"
        "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices at "
        "quadrature points.\n"
        "    GNeg (numpy.ndarray): `|# nodes per element| x |# dims * # quad.pts.|` shape "
        "function gradients at quadrature points.\n"
        "    element (EElement): Type of the finite element.\n"
        "    order (int): Order of the finite element.\n"
        "    dims (int): Number of spatial dimensions.\n"
        "Returns:\n"
        "    scipy.sparse matrix: `|# dims * # quad.pts.| x |# nodes|` gradient operator "
        "matrix.");
}

} // namespace fem
} // namespace py
} // namespace pbat