#include "Jacobian.h"

#include "Mesh.h"

#include <exception>
#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/Jacobian.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>

namespace pbat {
namespace py {
namespace fem {

void BindJacobian(nanobind::module_& m)
{
    namespace nb = nanobind;

    using TScalar = pbat::Scalar;
    using TIndex  = pbat::Index;

    m.def(
        "jacobian",
        [](nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> x,
           EElement eElement,
           int order) {
            auto const dims = x.rows();
            Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> J;
            auto constexpr kMaxDims = 9;
            if (dims < 1 or dims > kMaxDims)
            {
                throw std::invalid_argument(
                    fmt::format(
                        "Invalid number of dimensions, expected 1 <= dims <= {}, "
                        "but got {}",
                        kMaxDims,
                        dims));
            }
            pbat::common::ForRange<1, kMaxDims + 1>([&]<auto Dims>() {
                if (dims == Dims)
                {
                    ApplyToElement(eElement, order, [&]<class ElementType>() {
                        J = pbat::fem::Jacobian<ElementType>(
                            Xi.template topRows<ElementType::kDims>(),
                            x.template block<Dims, ElementType::kNodes>(0, 0));
                    });
                }
            });
            return J;
        },
        nb::arg("Xi"),
        nb::arg("x"),
        nb::arg("element"),
        nb::arg("order") = 1,
        "Computes the Jacobian matrix for a map x(Xi) at reference position Xi.\n\n"
        "Args:\n"
        "    Xi (numpy.ndarray): `|# ref dims| x 1` reference space coordinates.\n"
        "    x (numpy.ndarray): `|# dims| x |# elem nodes|` map coefficients.\n"
        "    element (EElement): Type of the finite element.\n"
        "    order (int): Order of the finite element.\n\n"
        "Returns:\n"
        "    numpy.ndarray: `|# dims| x |# ref dims|` Jacobian matrix");

    m.def(
        "determinant_of_jacobian",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
           EElement eElement,
           int order,
           int qOrder) {
            auto constexpr kMaxQuadratureOrder = 8;
            auto const dims                    = X.rows();
            auto constexpr kMaxDims            = 3;
            if (dims < 1 or dims > kMaxDims)
            {
                throw std::invalid_argument(
                    fmt::format(
                        "Invalid number of dimensions, expected 1 <= dims <= {}, "
                        "but got {}",
                        kMaxDims,
                        dims));
            }
            Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> detJe;
            ApplyToElementWithQuadrature<kMaxQuadratureOrder>(
                eElement,
                order,
                qOrder,
                [&]<class ElementType, auto QuadratureOrder>() {
                    pbat::common::ForRange<1, kMaxDims + 1>([&]<auto Dims>() {
                        if (dims == Dims)
                        {
                            detJe = pbat::fem::DeterminantOfJacobian<ElementType, QuadratureOrder>(
                                E.template topRows<ElementType::kNodes>(),
                                X.template topRows<Dims>());
                        }
                    });
                });
            return detJe;
        },
        nb::arg("E"),
        nb::arg("X"),
        nb::arg("element"),
        nb::arg("order")            = 1,
        nb::arg("quadrature_order") = 1,
        "Computes the determinant of the Jacobian matrix at element quadrature points.\n\n "
        " Args :\n "
        " E(numpy.ndarray) : `| # elem nodes | x | # elems |` element matrix.\n "
        " X(numpy.ndarray) : `| # dims | x | # nodes |` mesh nodal position matrix.\n "
        " element(EElement) : Type of the finite element.\n "
        " order(int) : Order of the finite element.\n "
        " quadrature_order(int) : Order of the quadrature rule.\n\n "
        " Returns :\n"
        " numpy.ndarray : `| # elem quad.pts.| x | # elems |` matrix of jacobian determinants");

    m.def(
        "determinant_of_jacobian_at",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
           nb::DRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
           EElement eElement,
           int order) {
            auto constexpr kMaxDims = 3;
            auto const dims         = X.rows();
            if (dims < 1 or dims > kMaxDims)
            {
                throw std::invalid_argument(
                    fmt::format(
                        "Invalid number of dimensions, expected 1 <= dims <= {}, "
                        "but got {}",
                        kMaxDims,
                        dims));
            }
            Eigen::Vector<TScalar, Eigen::Dynamic> detJe;
            ApplyToElement(eElement, order, [&]<class ElementType>() {
                pbat::common::ForRange<1, kMaxDims + 1>([&]<auto Dims>() {
                    if (dims == Dims)
                    {
                        detJe = pbat::fem::DeterminantOfJacobianAt<ElementType>(
                            E.template topRows<ElementType::kNodes>(),
                            X.template topRows<Dims>(),
                            eg,
                            Xi.template topRows<ElementType::kDims>());
                    }
                });
            });
            return detJe;
        },
        nb::arg("E"),
        nb::arg("X"),
        nb::arg("eg"),
        nb::arg("Xi"),
        nb::arg("element"),
        nb::arg("order") = 1,
        "Computes the determinant of the Jacobian matrix at evaluation points.\n\n"
        "Args:\n"
        "    E (numpy.ndarray): `|# elem nodes| x |# elems|` mesh element matrix.\n"
        "    X (numpy.ndarray): `|# dims| x |# nodes|` mesh nodal position matrix.\n"
        "    eg (numpy.ndarray): `|# eval.pts.|` element indices at evaluation points.\n "
        " Xi(numpy.ndarray) : `| # ref dims | x | # eval.pts.|` evaluation points in reference "
        "space.\n "
        " element(EElement) : Type of the finite element.\n "
        " order(int) : Order of the finite element.\n\n"
        "Returns:\n"
        "    numpy.ndarray: `|# eval.pts.| x 1` vector of jacobian determinants");

    m.def(
        "reference_positions",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
           nb::DRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xg,
           EElement eElement,
           int order,
           int maxIterations = 5,
           TScalar eps       = 1e-10) {
            auto constexpr kMaxDims = 3;
            auto const dims         = X.rows();
            if (dims < 1 or dims > kMaxDims)
            {
                throw std::invalid_argument(
                    fmt::format(
                        "Invalid number of dimensions, expected 1 <= dims <= {}, "
                        "but got {}",
                        kMaxDims,
                        dims));
            }
            Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> Xi;
            ApplyToElement(eElement, order, [&]<class ElementType>() {
                pbat::common::ForRange<1, kMaxDims + 1>([&]<auto Dims>() {
                    if (dims == Dims)
                    {
                        Xi = pbat::fem::ReferencePositions<ElementType>(
                            E.template topRows<ElementType::kNodes>(),
                            X.template topRows<Dims>(),
                            eg,
                            Xg.template topRows<Dims>(),
                            maxIterations,
                            eps);
                    }
                });
            });
            return Xi;
        },
        nb::arg("E"),
        nb::arg("X"),
        nb::arg("eg"),
        nb::arg("Xg"),
        nb::arg("element"),
        nb::arg("order")          = 1,
        nb::arg("max_iterations") = 5,
        nb::arg("eps")            = 1e-10,
        "Computes reference positions Xi such that X(Xi) = Xn for every point in Xg.\n\n"
        "Args:\n"
        "    E (numpy.ndarray): `|# elem nodes| x |# elems|` element matrix.\n"
        "    X (numpy.ndarray): `|# dims| x |# nodes|` nodal position matrix.\n"
        "    eg (numpy.ndarray): `|# eval.pts.|` indices of elements at evaluation "
        "points.\n"
        "    Xg (numpy.ndarray): `|# dims| x |# eval.pts.|` evaluation points in domain "
        "space.\n"
        "    element (EElement): Type of the finite element.\n"
        "    order (int): Order of the finite element.\n"
        "    max_iterations (int): Maximum number of Gauss-Newton iterations.\n"
        "    eps (float): Convergence tolerance.\n\n"
        "Returns:\n"
        "    numpy.ndarray: `|# element dims| x |# eval.pts.|` matrix of reference "
        "positions");
}

} // namespace fem
} // namespace py
} // namespace pbat
