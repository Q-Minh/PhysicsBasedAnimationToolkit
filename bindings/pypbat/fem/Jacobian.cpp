#include "Jacobian.h"

#include "Mesh.h"

#include <exception>
#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/Jacobian.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace fem {

void BindJacobian(pybind11::module& m)
{
    namespace pyb = pybind11;
    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        m.def(
            "jacobian",
            [](pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
               pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> x,
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
            pyb::arg("Xi"),
            pyb::arg("x"),
            pyb::arg("element"),
            pyb::arg("order") = 1,
            "Computes the Jacobian matrix for a map x(Xi) at reference position Xi.\n\n"
            "Args:\n"
            "    Xi (numpy.ndarray): `|# ref dims| x 1` reference space coordinates.\n"
            "    x (numpy.ndarray): `|# dims| x |# elem nodes|` map coefficients.\n"
            "    element (EElement): Type of the finite element.\n"
            "    order (int): Order of the finite element.\n\n"
            "Returns:\n"
            "    numpy.ndarray: `|# dims| x |# ref dims|` Jacobian matrix");

        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            //     m.def(
            //         "determinant_of_jacobian",
            //         [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>
            //         const> E,
            //            pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>
            //            const> X, EElement eElement, int order, int qOrder) {
            //             auto constexpr kMaxQuadratureOrder = 8;
            //             auto const dims                    = X.rows();
            //             auto constexpr kMaxDims            = 3;
            //             if (dims < 1 or dims > kMaxDims)
            //             {
            //                 throw std::invalid_argument(
            //                     fmt::format(
            //                         "Invalid number of dimensions, expected 1 <= dims <= {}, "
            //                         "but got {}",
            //                         kMaxDims,
            //                         dims));
            //             }
            //             Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> detJe;
            //             ApplyToElementWithQuadrature<kMaxQuadratureOrder>(
            //                 eElement,
            //                 order,
            //                 qOrder,
            //                 [&]<class ElementType, auto QuadratureOrder>() {
            //                     pbat::common::ForRange<1, kMaxDims + 1>([&]<auto Dims>() {
            //                         if (dims == Dims)
            //                         {
            //                             detJe = pbat::fem::
            //                                 DeterminantOfJacobian<ElementType, QuadratureOrder>(
            //                                     E.template topRows<ElementType::kNodes>(),
            //                                     X.template topRows<Dims>());
            //                         }
            //                     });
            //                 });
            //             return detJe;
            //         },
            //         pyb::arg("E"),
            //         pyb::arg("X"),
            //         pyb::arg("element"),
            //         pyb::arg("order")            = 1,
            //         pyb::arg("quadrature_order") = 1,
            //         "Computes the determinant of the Jacobian matrix at element quadrature
            //         points.\n\n" "Args:\n" "    E (numpy.ndarray): `|# elem nodes| x |# elems|`
            //         element matrix.\n" "    X (numpy.ndarray): `|# dims| x |# nodes|` mesh nodal
            //         position matrix.\n" "    element (EElement): Type of the finite
            //         element.\n" "    order (int): Order of the finite element.\n" "
            //         quadrature_order (int): Order of the quadrature rule.\n\n" "Returns:\n" "
            //         numpy.ndarray: `|# elem quad.pts.| x |# elems|` matrix of jacobian "
            //         "determinants");

            //     m.def(
            //         "determinant_of_jacobian_at",
            //         [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic>
            //         const> E,
            //            pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>
            //            const> X, pyb::EigenDRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
            //            pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>
            //            const> Xi, EElement eElement, int order) {
            //             auto constexpr kMaxDims = 3;
            //             auto const dims         = X.rows();
            //             if (dims < 1 or dims > kMaxDims)
            //             {
            //                 throw std::invalid_argument(
            //                     fmt::format(
            //                         "Invalid number of dimensions, expected 1 <= dims <= {}, "
            //                         "but got {}",
            //                         kMaxDims,
            //                         dims));
            //             }
            //             Eigen::Vector<TScalar, Eigen::Dynamic> detJe;
            //             ApplyToElement(eElement, order, [&]<class ElementType>() {
            //                 pbat::common::ForRange<1, kMaxDims + 1>([&]<auto Dims>() {
            //                     if (dims == Dims)
            //                     {
            //                         detJe = pbat::fem::DeterminantOfJacobianAt<ElementType>(
            //                             E.template topRows<ElementType::kNodes>(),
            //                             X.template topRows<Dims>(),
            //                             eg,
            //                             Xi.template topRows<ElementType::kDims>());
            //                     }
            //                 });
            //             });
            //             return detJe;
            //         },
            //         pyb::arg("E"),
            //         pyb::arg("X"),
            //         pyb::arg("eg"),
            //         pyb::arg("Xi"),
            //         pyb::arg("element"),
            //         pyb::arg("order") = 1,
            //         "Computes the determinant of the Jacobian matrix at evaluation points.\n\n"
            //         "Args:\n"
            //         "    E (numpy.ndarray): `|# elem nodes| x |# elems|` mesh element matrix.\n"
            //         "    X (numpy.ndarray): `|# dims| x |# nodes|` mesh nodal position matrix.\n"
            //         "    eg (numpy.ndarray): `|# eval.pts.|` element indices at evaluation
            //         points.\n" "    Xi (numpy.ndarray): `|# ref dims| x |# eval.pts.|` evaluation
            //         points in " "reference " "space.\n" "    element (EElement): Type of the
            //         finite element.\n" "    order (int): Order of the finite element.\n\n"
            //         "Returns:\n"
            //         "    numpy.ndarray: `|# eval.pts.| x 1` vector of jacobian determinants");

            // m.def(
            //     "reference_positions",
            //     [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
            //        pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const>
            //        X, pyb::EigenDRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
            //        pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const>
            //        Xg, EElement eElement, int order, int maxIterations = 5, TScalar eps       =
            //        1e-10) {
            //         auto constexpr kMaxDims = 3;
            //         auto const dims         = X.rows();
            //         if (dims < 1 or dims > kMaxDims)
            //         {
            //             throw std::invalid_argument(
            //                 fmt::format(
            //                     "Invalid number of dimensions, expected 1 <= dims <= {}, "
            //                     "but got {}",
            //                     kMaxDims,
            //                     dims));
            //         }
            //         Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> Xi;
            //         ApplyToElement(eElement, order, [&]<class ElementType>() {
            //             pbat::common::ForRange<1, kMaxDims + 1>([&]<auto Dims>() {
            //                 if (dims == Dims)
            //                 {
            //                     Xi = pbat::fem::ReferencePositions<ElementType>(
            //                         E.template topRows<ElementType::kNodes>(),
            //                         X.template topRows<Dims>(),
            //                         eg,
            //                         Xg.template topRows<Dims>(),
            //                         maxIterations,
            //                         eps);
            //                 }
            //             });
            //         });
            //         return Xi;
            //     },
            //     pyb::arg("E"),
            //     pyb::arg("X"),
            //     pyb::arg("eg"),
            //     pyb::arg("Xg"),
            //     pyb::arg("element"),
            //     pyb::arg("order")          = 1,
            //     pyb::arg("max_iterations") = 5,
            //     pyb::arg("eps")            = 1e-10,
            //     "Computes reference positions Xi such that X(Xi) = Xn for every point in Xg.\n\n"
            //     "Args:\n"
            //     "    E (numpy.ndarray): `|# elem nodes| x |# elems|` element matrix.\n"
            //     "    X (numpy.ndarray): `|# dims| x |# nodes|` nodal position matrix.\n"
            //     "    eg (numpy.ndarray): `|# eval.pts.|` indices of elements at evaluation "
            //     "points.\n"
            //     "    Xg (numpy.ndarray): `|# dims| x |# eval.pts.|` evaluation points in domain "
            //     "space.\n"
            //     "    element (EElement): Type of the finite element.\n"
            //     "    order (int): Order of the finite element.\n"
            //     "    max_iterations (int): Maximum number of Gauss-Newton iterations.\n"
            //     "    eps (float): Convergence tolerance.\n\n"
            //     "Returns:\n"
            //     "    numpy.ndarray: `|# element dims| x |# eval.pts.|` matrix of reference "
            //     "positions");
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat
