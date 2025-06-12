#include "LoadVector.h"

#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/LoadVector.h>
#include <pbat/fem/MeshQuadrature.h>
#include <pbat/fem/ShapeFunctions.h>
#include <pybind11/eigen.h>

namespace pbat::py::fem {

void BindLoadVector(pybind11::module& m)
{
    namespace pyb = pybind11;
    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            m.def(
                "load_vectors",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   Eigen::Index nNodes,
                   pyb::EigenDRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
                   pyb::EigenDRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> wg,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Neg,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Feg,
                   EElement eElement,
                   int order) {
                    ApplyToElement(eElement, order, [&]<class ElementType>() {
                        return pbat::fem::LoadVectors<ElementType>(
                            E.template topRows<ElementType::kNodes>(),
                            nNodes,
                            eg,
                            wg,
                            Neg.template topRows<ElementType::kNodes>(),
                            Feg);
                    });
                },
                pyb::arg("E"),
                pyb::arg("n_nodes"),
                pyb::arg("eg"),
                pyb::arg("wg"),
                pyb::arg("Neg"),
                pyb::arg("Feg"),
                pyb::arg("eElement"),
                pyb::arg("order") = 1,
                "Compute the load vector for a given FEM mesh.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# elem. nodes| x |# elements|` element matrix\n"
                "    n_nodes (int): Number of mesh nodes\n"
                "    eg (numpy.ndarray): `|# quad.pts.| x 1` elements associated with "
                "quadrature "
                "points\n"
                "    wg (numpy.ndarray): `|# quad.pts.| x 1` quadrature weights\n"
                "    Neg (numpy.ndarray): `|# elem. nodes| x |# quad.pts.|` nodal shape "
                "function "
                "values at "
                "quadrature points\n"
                "    Feg (numpy.ndarray): `|# dims| x |# quad.pts.|` load vector values at "
                "quadrature points\n"
                "    eElement (EElement): Element type\n"
                "    order (int): Order of the element shape functions\n"
                "Returns:\n"
                "    numpy.ndarray: `|# dims| x |# nodes|` matrix s.t. each output dimension's "
                "load vectors are stored in rows");

            m.def(
                "load_vectors",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
                   pyb::EigenDRef<Eigen::Vector<TScalar, Eigen::Dynamic> const> Fe,
                   EElement eElement,
                   int order,
                   int qOrder) {
                    ApplyToElementWithQuadrature<3>(
                        eElement,
                        order,
                        qOrder,
                        [&]<class ElementType, int QuadratureOrder>() {
                            auto const wg =
                                pbat::fem::MeshQuadratureWeights<ElementType, QuadratureOrder>(
                                    E.template topRows<ElementType::kNodes>(),
                                    X);
                            auto const eg = pbat::fem::MeshQuadratureElements<TIndex>(
                                static_cast<TIndex>(E.cols()),
                                static_cast<TIndex>(wg.cols()));
                            auto const nElements = E.cols();
                            auto const nQuadPts  = wg.size();
                            auto const Ng        = pbat::fem::
                                ElementShapeFunctions<ElementType, QuadratureOrder, TScalar>();
                            auto const Neg = Ng.replicate(1, nElements);
                            auto const Feg = Fe.replicate(1, nQuadPts);
                            return pbat::fem::LoadVectors<ElementType>(
                                E.template topRows<ElementType::kNodes>(),
                                X.cols(),
                                eg.reshaped(),
                                wg.reshaped(),
                                Neg,
                                Feg);
                        });
                },
                pyb::arg("E"),
                pyb::arg("X"),
                pyb::arg("Fe"),
                pyb::arg("eElement"),
                pyb::arg("order")  = 1,
                pyb::arg("qOrder") = 1,
                "Compute the load vector for a given FEM mesh.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# elem. nodes| x |# elements|` element matrix\n"
                "    X (numpy.ndarray): `|# dims| x |# nodes|` mesh nodes\n"
                "    Fe (numpy.ndarray): `|# dims| x 1` uniform external load\n"
                "    eElement (EElement): Element type\n"
                "    order (int): Order of the element shape functions\n"
                "    qOrder (int): Order of the quadrature rule\n"
                "Returns:\n"
                "    numpy.ndarray: `|# dims| x |# nodes|` matrix s.t. each output dimension's "
                "load vectors are stored in rows");
        });
    });
}

} // namespace pbat::py::fem