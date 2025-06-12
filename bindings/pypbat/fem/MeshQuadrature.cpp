#include "MeshQuadrature.h"

#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/MeshQuadrature.h>
#include <pybind11/eigen.h>

namespace pbat::py::fem {

void BindMeshQuadrature([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            m.def(
                "mesh_quadrature_weights",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
                   EElement eElement,
                   int order,
                   int qOrder) {
                    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> wg;
                    ApplyToElementWithQuadrature<8>(
                        eElement,
                        order,
                        qOrder,
                        [&]<class ElementType, auto QuadratureOrder>() {
                            wg = pbat::fem::MeshQuadratureWeights<ElementType, QuadratureOrder>(
                                E.template topRows<ElementType::kNodes>(),
                                X);
                        });
                    return wg;
                },
                pyb::arg("E"),
                pyb::arg("X"),
                pyb::arg("element"),
                pyb::arg("order")            = 1,
                pyb::arg("quadrature_order") = 1,
                "Compute mesh quadrature weights including Jacobian determinants.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# elem nodes| x |# elems|` mesh element matrix.\n"
                "    X (numpy.ndarray): `|# dims| x |# nodes|` mesh nodal position matrix.\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    quadrature_order (int): Order of the quadrature rule.\n\n"
                "Returns:\n"
                "    numpy.ndarray: `|# quad.pts.| x |# elements|` matrix of quadrature weights.");

            m.def(
                "mesh_quadrature_elements",
                [](TIndex nElements, TIndex nQuadPtsPerElement) {
                    return pbat::fem::MeshQuadratureElements<TIndex>(nElements, nQuadPtsPerElement)
                        .eval();
                },
                pyb::arg("n_elements"),
                pyb::arg("n_quad_pts_per_element"),
                "Compute element indices for each quadrature point.\n\n"
                "Args:\n"
                "    n_elements (int): Number of elements.\n"
                "    n_quad_pts_per_element (int): Number of quadrature points per element.\n\n"
                "Returns:\n"
                "    numpy.ndarray: `|# quad.pts.| x |# elems|` matrix of element indices.");

            m.def(
                "mesh_quadrature_elements",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const>
                       wg) { return pbat::fem::MeshQuadratureElements(E, wg).eval(); },
                pyb::arg("E"),
                pyb::arg("wg"),
                "Compute element indices for each quadrature point from element matrix and "
                "weights.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# elem nodes| x |# elems|` mesh element matrix.\n"
                "    wg (numpy.ndarray): `|# quad.pts.| x |# elems|` mesh quadrature weights "
                "matrix.\n\n"
                "Returns:\n"
                "    numpy.ndarray: `|# quad.pts. * # elems|` flattened vector of element "
                "indices.");

            m.def(
                "mesh_reference_quadrature_points",
                [](TIndex nElements, EElement eElement, int order, int qOrder) {
                    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> Xi;
                    ApplyToElementWithQuadrature<
                        8>(eElement, order, qOrder, [&]<class ElementType, auto QuadratureOrder>() {
                        Xi = pbat::fem::
                            MeshReferenceQuadraturePoints<ElementType, QuadratureOrder, TScalar>(
                                nElements);
                    });
                    return Xi;
                },
                pyb::arg("n_elements"),
                pyb::arg("element"),
                pyb::arg("order")            = 1,
                pyb::arg("quadrature_order") = 1,
                "Compute quadrature points in reference element space for all elements.\n\n"
                "Args:\n"
                "    n_elements (int): Number of elements.\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    quadrature_order (int): Order of the quadrature rule.\n\n"
                "Returns:\n"
                "    numpy.ndarray: `|# ref. dims| x |# quad.pts. * # elems|` matrix of quadrature "
                "points in reference space.");
        });
    });
}

} // namespace pbat::py::fem