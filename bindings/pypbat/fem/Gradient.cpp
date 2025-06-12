#include "Gradient.h"

#include "Mesh.h"

#include <pbat/fem/Gradient.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace fem {

void BindGradient([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            m.def(
                "gradient_matrix",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   TIndex nNodes,
                   pyb::EigenDRef<Eigen::Vector<TIndex, Eigen::Dynamic> const> eg,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const>
                       GNeg,
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
                pyb::arg("E"),
                pyb::arg("n_nodes"),
                pyb::arg("eg"),
                pyb::arg("GNeg"),
                pyb::arg("element"),
                pyb::arg("order") = 1,
                pyb::arg("dims")  = 3,
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
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat