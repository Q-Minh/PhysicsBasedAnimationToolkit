#include "Mass.h"

#include "Mesh.h"

#include <pbat/fem/Mass.h>
#include <pbat/fem/MeshQuadrature.h>
#include <pbat/fem/ShapeFunctions.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace fem {

void BindMass([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            m.def(
                "element_mass_matrices",
                [](pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Neg,
                   pyb::EigenDRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> wg,
                   pyb::EigenDRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> rhog,
                   EElement eElement,
                   int order) {
                    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> Meg;
                    ApplyToElement(eElement, order, [&]<class ElementType>() {
                        Meg = pbat::fem::ElementMassMatrices<ElementType>(
                            Neg.template topRows<ElementType::kNodes>(),
                            wg,
                            rhog);
                    });
                    return Meg;
                },
                pyb::arg("Neg"),
                pyb::arg("wg"),
                pyb::arg("rhog"),
                pyb::arg("element"),
                pyb::arg("order") = 1,
                "Compute element mass matrices for all quadrature points.\n\n"
                "Args:\n"
                "    Neg (numpy.ndarray): `|# nodes per element| x |# quad.pts.|` shape function "
                "matrix at all quadrature points.\n"
                "    wg (numpy.ndarray): `|# quad.pts.| x 1` quadrature weights (including "
                "Jacobian determinant).\n"
                "    rhog (numpy.ndarray): `|# quad.pts.| x 1` mass density at quadrature points.\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n\n"
                "Returns:\n"
                "    numpy.ndarray: `|# elem nodes| x |# elem nodes * # quad.pts.|` matrix of "
                "stacked element mass matrices.");

            m.def(
                "mass_matrix",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   TIndex nNodes,
                   pyb::EigenDRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
                   pyb::EigenDRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> wg,
                   pyb::EigenDRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> rhog,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Neg,
                   int dims,
                   EElement eElement,
                   int order,
                   int spatialDims) {
                    Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> M;
                    ApplyToElementInDims(
                        eElement,
                        order,
                        spatialDims,
                        [&]<class ElementType, int Dims>() {
                            M = pbat::fem::MassMatrix<ElementType, Dims, Eigen::RowMajor>(
                                E.template topRows<ElementType::kNodes>(),
                                nNodes,
                                eg,
                                wg,
                                rhog,
                                Neg.template topRows<ElementType::kNodes>(),
                                dims);
                        });
                    return M;
                },
                pyb::arg("E"),
                pyb::arg("n_nodes"),
                pyb::arg("eg"),
                pyb::arg("wg"),
                pyb::arg("rhog"),
                pyb::arg("Neg"),
                pyb::arg("dims") = 1,
                pyb::arg("element"),
                pyb::arg("order")        = 1,
                pyb::arg("spatial_dims") = 3,
                "Construct the mass matrix operator's sparse matrix representation.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
                "elements.\n"
                "    n_nodes (int): Number of mesh nodes.\n"
                "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices at "
                "quadrature points.\n"
                "    wg (numpy.ndarray): `|# quad.pts.| x 1` vector of quadrature weights "
                "(including Jacobian determinants).\n"
                "    rhog (numpy.ndarray): `|# quad.pts.| x 1` vector of density at quadrature "
                "points.\n"
                "    Neg (numpy.ndarray): `|# nodes per element| x |# quad.pts.|` shape functions "
                "at quadrature points.\n"
                "    dims (int): Dimensionality of the image of the FEM function space (default: "
                "1).\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    spatial_dims (int): Number of spatial dimensions.\n"
                "Returns:\n"
                "    scipy.sparse matrix: `|# nodes * dims| x |# nodes * dims|` mass matrix "
                "operator.");

            m.def(
                "mass_matrix",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
                   TScalar rho,
                   int dims,
                   EElement eElement,
                   int order) {
                    Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> M;
                    auto const spatialDims = static_cast<int>(X.rows());
                    ApplyToElementInDims(
                        eElement,
                        order,
                        spatialDims,
                        [&]<class ElementType, int Dims>() {
                            auto constexpr kQuadOrder = 2 * ElementType::kOrder;
                            auto const wg =
                                pbat::fem::MeshQuadratureWeights<ElementType, kQuadOrder>(
                                    E.template topRows<ElementType::kNodes>(),
                                    X.template topRows<Dims>());
                            auto const eg = pbat::fem::MeshQuadratureElements(
                                E.template topRows<ElementType::kNodes>(),
                                wg);
                            auto const rhog =
                                Eigen::Vector<TScalar, Eigen::Dynamic>::Constant(wg.size(), rho);
                            auto const Ng = pbat::fem::
                                ElementShapeFunctions<ElementType, kQuadOrder, TScalar>();
                            auto const Neg = Ng.replicate(1, E.cols());
                            M = pbat::fem::MassMatrix<ElementType, Dims, Eigen::RowMajor>(
                                E.template topRows<ElementType::kNodes>(),
                                X.cols(),
                                eg.reshaped(),
                                wg.reshaped(),
                                rhog,
                                Neg.template topRows<ElementType::kNodes>(),
                                dims);
                        });
                    return M;
                },
                pyb::arg("E"),
                pyb::arg("X"),
                pyb::arg("rho")  = TScalar(1e3),
                pyb::arg("dims") = 1,
                pyb::arg("element"),
                pyb::arg("order") = 1,
                "Construct the mass matrix operator's sparse matrix representation.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
                "elements.\n"
                "    X (numpy.ndarray): `|# nodes| x |# spatial dims|` matrix of nodal "
                "coordinates.\n"
                "    rho (float): Density of the material (default: 1e3).\n"
                "    dims (int): Dimensionality of the image of the FEM function space (default: "
                "1).\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "Returns:\n"
                "    scipy.sparse matrix: `|# nodes * dims| x |# nodes * dims|` mass matrix "
                "operator.");

            m.def(
                "mass_matrix",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   TIndex nNodes,
                   pyb::EigenDRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Meg,
                   int dims,
                   EElement eElement,
                   int order,
                   int spatialDims) {
                    Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> M;
                    ApplyToElementInDims(
                        eElement,
                        order,
                        spatialDims,
                        [&]<class ElementType, int Dims>() {
                            M = pbat::fem::MassMatrix<ElementType, Dims, Eigen::RowMajor>(
                                E.template topRows<ElementType::kNodes>(),
                                nNodes,
                                eg,
                                Meg.template topRows<ElementType::kNodes>(),
                                dims);
                        });
                    return M;
                },
                pyb::arg("E"),
                pyb::arg("n_nodes"),
                pyb::arg("eg"),
                pyb::arg("Meg"),
                pyb::arg("dims") = 1,
                pyb::arg("element"),
                pyb::arg("order")        = 1,
                pyb::arg("spatial_dims") = 3,
                "Construct the mass matrix operator's sparse matrix representation from "
                "precomputed element mass matrices.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
                "elements.\n"
                "    n_nodes (int): Number of mesh nodes.\n"
                "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices at "
                "quadrature points.\n"
                "    Meg (numpy.ndarray): `|# nodes per element| x |# nodes per element * # "
                "quad.pts.|` precomputed element mass matrices.\n"
                "    dims (int): Dimensionality of the image of the FEM function space (default: "
                "1).\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    spatial_dims (int): Number of spatial dimensions.\n"
                "Returns:\n"
                "    scipy.sparse matrix: `|# nodes * dims| x |# nodes * dims|` mass matrix "
                "operator.");

            m.def(
                "lumped_mass_matrix",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   TIndex nNodes,
                   pyb::EigenDRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
                   pyb::EigenDRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> wg,
                   pyb::EigenDRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> rhog,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Neg,
                   int dims,
                   EElement eElement,
                   int order,
                   int spatialDims) {
                    Eigen::Vector<TScalar, Eigen::Dynamic> m;
                    ApplyToElementInDims(
                        eElement,
                        order,
                        spatialDims,
                        [&]<class ElementType, int Dims>() {
                            m = pbat::fem::LumpedMassMatrix<ElementType, Dims>(
                                E.template topRows<ElementType::kNodes>(),
                                nNodes,
                                eg,
                                wg,
                                rhog,
                                Neg.template topRows<ElementType::kNodes>(),
                                dims);
                        });
                    return m;
                },
                pyb::arg("E"),
                pyb::arg("n_nodes"),
                pyb::arg("eg"),
                pyb::arg("wg"),
                pyb::arg("rhog"),
                pyb::arg("Neg"),
                pyb::arg("dims") = 1,
                pyb::arg("element"),
                pyb::arg("order")        = 1,
                pyb::arg("spatial_dims") = 3,
                "Compute lumped mass matrix's diagonal vector.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
                "elements.\n"
                "    n_nodes (int): Number of mesh nodes.\n"
                "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices at "
                "quadrature points.\n"
                "    wg (numpy.ndarray): `|# quad.pts.| x 1` vector of quadrature weights "
                "(including Jacobian determinants).\n"
                "    rhog (numpy.ndarray): `|# quad.pts.| x 1` vector of density at quadrature "
                "points.\n"
                "    Neg (numpy.ndarray): `|# nodes per element| x |# quad.pts.|` shape functions "
                "at quadrature points.\n"
                "    dims (int): Dimensionality of the image of the FEM function space (default: "
                "1).\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    spatial_dims (int): Number of spatial dimensions.\n\n"
                "Returns:\n"
                "    numpy.ndarray: `|# nodes * dims| x 1` vector of lumped masses.");

            m.def(
                "lumped_mass_matrix",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
                   TScalar rho,
                   int dims,
                   EElement eElement,
                   int order) {
                    Eigen::Vector<TScalar, Eigen::Dynamic> m;
                    auto const spatialDims = static_cast<int>(X.rows());
                    ApplyToElementInDims(
                        eElement,
                        order,
                        spatialDims,
                        [&]<class ElementType, int Dims>() {
                            auto constexpr kQuadOrder = 2 * ElementType::kOrder;
                            auto const wg =
                                pbat::fem::MeshQuadratureWeights<ElementType, kQuadOrder>(
                                    E.template topRows<ElementType::kNodes>(),
                                    X.template topRows<Dims>());
                            auto const eg = pbat::fem::MeshQuadratureElements(
                                E.template topRows<ElementType::kNodes>(),
                                wg);
                            auto const rhog =
                                Eigen::Vector<TScalar, Eigen::Dynamic>::Constant(wg.size(), rho);
                            auto const Ng = pbat::fem::
                                ElementShapeFunctions<ElementType, kQuadOrder, TScalar>();
                            auto const Neg = Ng.replicate(1, E.cols());
                            m              = pbat::fem::LumpedMassMatrix<ElementType, Dims>(
                                E.template topRows<ElementType::kNodes>(),
                                X.cols(),
                                eg.reshaped(),
                                wg.reshaped(),
                                rhog,
                                Neg.template topRows<ElementType::kNodes>(),
                                dims);
                        });
                    return m;
                },
                pyb::arg("E"),
                pyb::arg("X"),
                pyb::arg("rho")  = TScalar(1e3),
                pyb::arg("dims") = 1,
                pyb::arg("element"),
                pyb::arg("order") = 1,
                "Compute lumped mass matrix's diagonal vector.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
                "elements.\n"
                "    n_nodes (int): Number of mesh nodes.\n"
                "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices at "
                "quadrature points.\n"
                "    wg (numpy.ndarray): `|# quad.pts.| x 1` vector of quadrature weights "
                "(including Jacobian determinants).\n"
                "    rhog (numpy.ndarray): `|# quad.pts.| x 1` vector of density at quadrature "
                "points.\n"
                "    Neg (numpy.ndarray): `|# nodes per element| x |# quad.pts.|` shape functions "
                "at quadrature points.\n"
                "    dims (int): Dimensionality of the image of the FEM function space (default: "
                "1).\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "Returns:\n"
                "    numpy.ndarray: `|# nodes * dims| x 1` vector of lumped masses.");

            m.def(
                "lumped_mass_matrix",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   TIndex nNodes,
                   pyb::EigenDRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Meg,
                   int dims,
                   EElement eElement,
                   int order,
                   int spatialDims) {
                    Eigen::Vector<TScalar, Eigen::Dynamic> m;
                    ApplyToElementInDims(
                        eElement,
                        order,
                        spatialDims,
                        [&]<class ElementType, int Dims>() {
                            m = pbat::fem::LumpedMassMatrix<ElementType, Dims>(
                                E.template topRows<ElementType::kNodes>(),
                                nNodes,
                                eg,
                                Meg.template topRows<ElementType::kNodes>(),
                                dims);
                        });
                    return m;
                },
                pyb::arg("E"),
                pyb::arg("n_nodes"),
                pyb::arg("eg"),
                pyb::arg("Meg"),
                pyb::arg("dims") = 1,
                pyb::arg("element"),
                pyb::arg("order")        = 1,
                pyb::arg("spatial_dims") = 3,
                "Compute lumped mass vector from precomputed element mass matrices.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# nodes per element| x |# elements|` matrix of mesh "
                "elements.\n"
                "    n_nodes (int): Number of mesh nodes.\n"
                "    eg (numpy.ndarray): `|# quad.pts.| x 1` vector of element indices at "
                "quadrature points.\n"
                "    Meg (numpy.ndarray): `|# nodes per element| x |# nodes per element * # "
                "quad.pts.|` precomputed element mass matrices.\n"
                "    dims (int): Dimensionality of the image of the FEM function space (default: "
                "1).\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    spatial_dims (int): Number of spatial dimensions.\n\n"
                "Returns:\n"
                "    numpy.ndarray: `|# nodes * dims| x 1` vector of lumped masses.");
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat