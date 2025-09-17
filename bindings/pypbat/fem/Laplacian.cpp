#include "Laplacian.h"

#include "Mesh.h"

#include <pbat/fem/Laplacian.h>
#include <pbat/fem/MeshQuadrature.h>
#include <pbat/fem/ShapeFunctions.h>
#include <nanobind/eigen/dense.h>

namespace pbat {
namespace py {
namespace fem {

void BindLaplacian([[maybe_unused]] nanobind::module_& m)
{
    namespace nb = nanobind;

    using TScalar = pbat::Scalar;
    using TIndex  = pbat::Index;
    
    m.def(
        "laplacian_matrix",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           TIndex nNodes,
           nb::DRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
           nb::DRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> wg,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> GNeg,
           int dims,
           EElement eElement,
           int order,
           int spatialDims) {
            Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> L;
            ApplyToElementInDims(eElement, order, spatialDims, [&]<class ElementType, int Dims>() {
                L = pbat::fem::LaplacianMatrix<ElementType, Dims, Eigen::RowMajor>(
                    E.template topRows<ElementType::kNodes>(),
                    nNodes,
                    eg,
                    wg,
                    GNeg.template topRows<ElementType::kNodes>(),
                    dims);
            });
            return L;
        },
        nb::arg("E"),
        nb::arg("n_nodes"),
        nb::arg("eg"),
        nb::arg("wg"),
        nb::arg("GNeg"),
        nb::arg("dims") = 1,
        nb::arg("element"),
        nb::arg("order")        = 1,
        nb::arg("spatial_dims") = 3,
        "Construct the Laplacian operator's sparse matrix representation.\n\n"
        "Args:\n"
        "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
        "elements.\n"
        "    n_nodes (int): Number of mesh nodes.\n"
        "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices at "
        "quadrature "
        "points.\n"
        "    wg (numpy.ndarray): `|# quad.pts.| x 1` vector of quadrature weights.\n"
        "    GNeg (numpy.ndarray): `|# nodes per element| x |# dims * # quad.pts.|` shape "
        "function gradients at quadrature points.\n"
        "    dims (int): Dimensionality of the image of the FEM function space (default: "
        "1).\n"
        "    element (EElement): Type of the finite element.\n"
        "    order (int): Order of the finite element.\n"
        "    spatial_dims (int): Number of spatial dimensions.\n"
        "Returns:\n"
        "    scipy.sparse matrix: `|# nodes * dims| x |# nodes * dims|` Laplacian operator "
        "matrix.");

    m.def(
        "laplacian_matrix",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
           int dims,
           EElement eElement,
           int order) {
            Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> L;
            auto const spatialDims = static_cast<int>(X.rows());
            ApplyToElementInDims(eElement, order, spatialDims, [&]<class ElementType, int Dims>() {
                auto constexpr kQuadOrder = std::max(1, 2 * (ElementType::kOrder - 1));
                auto const wg = pbat::fem::MeshQuadratureWeights<ElementType, kQuadOrder>(E, X);
                auto const eg = pbat::fem::MeshQuadratureElements(
                    E.template topRows<ElementType::kNodes>(),
                    wg);
                auto const GNeg = pbat::fem::ShapeFunctionGradients<ElementType, Dims, kQuadOrder>(
                    E.template topRows<ElementType::kNodes>(),
                    X.template topRows<Dims>());
                L = pbat::fem::LaplacianMatrix<ElementType, Dims, Eigen::RowMajor>(
                    E.template topRows<ElementType::kNodes>(),
                    X.cols(),
                    eg.reshaped(),
                    wg.reshaped(),
                    GNeg.template topRows<ElementType::kNodes>(),
                    dims);
            });
            return L;
        },
        nb::arg("E"),
        nb::arg("X"),
        nb::arg("dims") = 1,
        nb::arg("element"),
        nb::arg("order") = 1,
        "Construct the Laplacian operator's sparse matrix representation.\n\n"
        "Args:\n"
        "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
        "elements.\n"
        "    X (numpy.ndarray): `|# dims * # nodes| x |# nodes|` matrix of node "
        "positions.\n"
        "    dims (int): Dimensionality of the image of the FEM function space (default: "
        "1).\n"
        "    element (EElement): Type of the finite element.\n"
        "    order (int): Order of the finite element.\n"
        "Returns:\n"
        "    scipy.sparse matrix: `|# nodes * dims| x |# nodes * dims|` Laplacian operator "
        "matrix.");
}

} // namespace fem
} // namespace py
} // namespace pbat