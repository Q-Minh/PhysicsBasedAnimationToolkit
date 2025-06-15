#include "ShapeFunctions.h"

#include "Mesh.h"

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/ShapeFunctions.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <variant>

namespace pbat {
namespace py {
namespace fem {

void BindShapeFunctions(pybind11::module& m)
{
    namespace pyb = pybind11;

    m.def(
        "element_shape_functions",
        [](EElement eElement, int order, int quadratureOrder, pyb::object dtypeIn) {
            using FloatMatrixType  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
            using DoubleMatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
            using ReturnType       = std::variant<FloatMatrixType, DoubleMatrixType>;
            ReturnType N;
            auto dtype = pyb::dtype::from_args(dtypeIn);
            if (dtype.is(pyb::dtype::of<float>()))
            {
                ApplyToElementWithQuadrature<6>(
                    eElement,
                    order,
                    quadratureOrder,
                    [&]<class ElementType, auto QuadratureOrder>() {
                        N = FloatMatrixType(
                            pbat::fem::
                                ElementShapeFunctions<ElementType, QuadratureOrder, float>());
                    });
            }
            else if (dtype.is(pyb::dtype::of<double>()))
            {
                ApplyToElementWithQuadrature<6>(
                    eElement,
                    order,
                    quadratureOrder,
                    [&]<class ElementType, auto QuadratureOrder>() {
                        N = DoubleMatrixType(
                            pbat::fem::
                                ElementShapeFunctions<ElementType, QuadratureOrder, double>());
                    });
            }
            else
            {
                throw std::runtime_error(
                    "Unsupported dtype for element shape functions: " +
                    dtypeIn.attr("name").cast<std::string>());
            }
            return N;
        },
        pyb::arg("element"),
        pyb::arg("order")            = 1,
        pyb::arg("quadrature_order") = 1,
        pyb::arg("dtype")            = pyb::dtype::of<float>(),
        "Compute an element's shape functions at the quadrature points of the requested "
        "quadrature rule\n\n"
        "Args:\n"
        "    element (EElement): The type of the element (e.g., Line, Triangle, "
        "Quadrilateral, Tetrahedron, Hexahedron)\n"
        "    order (int): The polynomial order of the shape functions (default: 1)\n"
        "    quadrature_order (int): The order of the quadrature rule to use (default: 1)\n\n"
        "    dtype (numpy.dtype): The data type of the output array (default: float32)\n"
        "Returns:\n"
        "    numpy.ndarray: `|# nodes| x |# quad.pts.|` shape function matrix");

    m.def(
        "shape_functions",
        [](std::int64_t nElements,
           EElement eElement,
           int order,
           int quadratureOrder,
           pyb::object dtypeIn) {
            using FloatMatrixType  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
            using DoubleMatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
            using ReturnType       = std::variant<FloatMatrixType, DoubleMatrixType>;
            ReturnType N;
            auto dtype = pyb::dtype::from_args(dtypeIn);
            if (dtype.is(pyb::dtype::of<float>()))
            {
                ApplyToElementWithQuadrature<6>(
                    eElement,
                    order,
                    quadratureOrder,
                    [&]<class ElementType, auto QuadratureOrder>() {
                        N = FloatMatrixType(
                            pbat::fem::ElementShapeFunctions<ElementType, QuadratureOrder, float>()
                                .replicate(1, nElements));
                    });
            }
            else if (dtype.is(pyb::dtype::of<double>()))
            {
                ApplyToElementWithQuadrature<6>(
                    eElement,
                    order,
                    quadratureOrder,
                    [&]<class ElementType, auto QuadratureOrder>() {
                        N = DoubleMatrixType(
                            pbat::fem::ElementShapeFunctions<ElementType, QuadratureOrder, double>()
                                .replicate(1, nElements));
                    });
            }
            else
            {
                throw std::runtime_error(
                    "Unsupported dtype for shape functions: " +
                    dtypeIn.attr("name").cast<std::string>());
            }
            return N;
        },
        pyb::arg("n_elements"),
        pyb::arg("element"),
        pyb::arg("order")            = 1,
        pyb::arg("quadrature_order") = 1,
        pyb::arg("dtype")            = pyb::dtype::of<float>(),
        "Compute an element's shape functions at the quadrature points of the requested "
        "quadrature rule\n\n"
        "Args:\n"
        "    n_elements (int): The number of elements in the mesh.\n"
        "    element (EElement): The type of the element (e.g., Line, Triangle, "
        "Quadrilateral, Tetrahedron, Hexahedron)\n"
        "    order (int): The polynomial order of the shape functions (default: 1)\n"
        "    quadrature_order (int): The order of the quadrature rule to use (default: "
        "1)\n\n"
        "    dtype (numpy.dtype): The data type of the output array (default: float32)\n"
        "Returns:\n"
        "    numpy.ndarray: `|# nodes| x |# quad.pts.|` shape function matrix");

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            m.def(
                "shape_function_matrix",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   TIndex nNodes,
                   EElement eElement,
                   int order,
                   int qOrder) {
                    Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> N;
                    ApplyToElementWithQuadrature<
                        6>(eElement, order, qOrder, [&]<class ElementType, int QuadratureOrder>() {
                        N = pbat::fem::ShapeFunctionMatrix<ElementType, QuadratureOrder, TScalar>(
                            E.template topRows<ElementType::kNodes>(),
                            nNodes);
                    });
                    return N;
                },
                pyb::arg("E"),
                pyb::arg("n_nodes"),
                pyb::arg("element"),
                pyb::arg("order")            = 1,
                pyb::arg("quadrature_order") = 1,
                "Constructs a shape function matrix N for a given mesh, i.e. at the "
                "element quadrature points.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# elem. nodes| x |# elements|` array of element node "
                "indices.\n"
                "    n_nodes (int): Number of nodes in the mesh.\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    quadrature_order (int): Order of the quadrature rule to use.\n\n"
                "Returns:\n"
                "    scipy.sparse.csr_matrix: `|# elements * # quad.pts.| x |# nodes|` shape "
                "function matrix");
        });
    });

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            m.def(
                "shape_function_matrix_at",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   TIndex nNodes,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
                   EElement eElement,
                   int order) {
                    Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> N;
                    ApplyToElement(eElement, order, [&]<class ElementType>() {
                        N = pbat::fem::ShapeFunctionMatrixAt<ElementType>(
                            E.template topRows<ElementType::kNodes>(),
                            nNodes,
                            Xi.template topRows<ElementType::kDims>());
                    });
                    return N;
                },
                pyb::arg("E"),
                pyb::arg("n_nodes"),
                pyb::arg("Xi"),
                pyb::arg("element"),
                pyb::arg("order") = 1,
                "Constructs a shape function matrix N for a given mesh, i.e. at the element "
                "quadrature points.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# elem. nodes| x |# elements|` array of element node "
                "indices.\n"
                "    n_nodes (int): Number of nodes in the mesh.\n"
                "    Xi (numpy.ndarray): `|# dims| x |# quad. pts.|` array of quadrature points in "
                "reference space.\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n\n"
                "Returns:\n"
                "    scipy.sparse.csr_matrix: `|# quad.pts.| x |# nodes|` shape function matrix at "
                "the quadrature points Xi");
        });
    });

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        m.def(
            "shape_functions_at",
            [](pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
               EElement eElement,
               int order) {
                Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> N;
                ApplyToElement(eElement, order, [&]<class ElementType>() {
                    N = pbat::fem::ShapeFunctionsAt<ElementType>(
                        Xi.template topRows<ElementType::kDims>());
                });
                return N;
            },
            pyb::arg("Xi"),
            pyb::arg("element"),
            pyb::arg("order") = 1,
            "Compute shape functions at the given reference positions.\n\n"
            "Args:\n"
            "    Xi (numpy.ndarray): `|# reference dims| x |# quad. pts.|` evaluation points in "
            "reference space.\n"
            "    element (EElement): Type of the finite element.\n"
            "    order (int): Order of the finite element.\n\n"
            "Returns:\n"
            "    numpy.ndarray: `|# element nodes| x |# eval.pts.|` matrix of nodal shape "
            "functions "
            "values at the evaluation points Xi");
    });

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        m.def(
            "element_shape_function_gradients",
            [](pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
               pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
               EElement eElement,
               int order) {
                Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> GN;
                auto dims = static_cast<int>(X.rows());
                ApplyToElementInDims(eElement, order, dims, [&]<class ElementType, int Dims>() {
                    GN = pbat::fem::ElementShapeFunctionGradients<ElementType>(
                        Xi.template topRows<ElementType::kNodes>(),
                        X.template topRows<Dims>());
                });
                return GN;
            },
            pyb::arg("Xi"),
            pyb::arg("X"),
            pyb::arg("element"),
            pyb::arg("order") = 1,
            "Compute gradients of an element's shape functions at the given reference point, given "
            "the vertices (not nodes) of the element.\n\n"
            "Args:\n"
            "    Xi (numpy.ndarray): `|# reference dims|` evaluation point in reference space.\n"
            "    X (numpy.ndarray): `|# dims| x |# element vertices|` array of element vertices.\n"
            "    element (EElement): The type of the element.\n"
            "    order (int): The polynomial order of the shape functions (default: 1)\n"
            "Returns:\n"
            "    numpy.ndarray: `|# element nodes| x |# dims|` matrix of shape function gradients "
            "at the evaluation point Xi");
    });

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            m.def(
                "shape_function_gradients",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
                   EElement eElement,
                   int order,
                   int dims,
                   int qOrder) {
                    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> GNeg;
                    ApplyToElementInDimsWithQuadrature<6>(
                        eElement,
                        order,
                        dims,
                        qOrder,
                        [&]<class ElementType, int Dims, int QuadratureOrder>() {
                            GNeg = pbat::fem::
                                ShapeFunctionGradients<ElementType, Dims, QuadratureOrder>(
                                    E.template topRows<ElementType::kNodes>(),
                                    X.template topRows<Dims>());
                        });
                    return GNeg;
                },
                pyb::arg("E"),
                pyb::arg("X"),
                pyb::arg("element"),
                pyb::arg("order")            = 1,
                pyb::arg("dims")             = 3,
                pyb::arg("quadrature_order") = 1,
                "Computes nodal shape function gradients at each element quadrature points.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# elem. nodes| x |# elements|` array of elements.\n"
                "    X (numpy.ndarray): `|# dims| x |# element vertices|` array of nodes.\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n"
                "    quadrature_order (int): Order of the quadrature rule to use.\n\n"
                "Returns:\n"
                "    numpy.ndarray: `|# elem. nodes| x |# dims * #elems * # elem.quad. pts.|` "
                "matrix of shape "
                "function gradients at each element quadrature point.");
        });
    });

    pbat::common::ForTypes<float, double>([&]<class TScalar>() {
        pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<class TIndex>() {
            m.def(
                "shape_function_gradients_at",
                [](pyb::EigenDRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
                   pyb::EigenDRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
                   EElement eElement,
                   int order,
                   int dims) {
                    Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> GNeg;
                    ApplyToElementInDims(eElement, order, dims, [&]<class ElementType, int Dims>() {
                        GNeg = pbat::fem::ShapeFunctionGradientsAt<ElementType, Dims>(
                            E.template topRows<ElementType::kNodes>(),
                            X.template topRows<Dims>(),
                            Xi.template topRows<ElementType::kDims>());
                    });
                    return GNeg;
                },
                pyb::arg("E"),
                pyb::arg("X"),
                pyb::arg("Xi"),
                pyb::arg("element"),
                pyb::arg("order") = 1,
                pyb::arg("dims")  = 3,
                "Computes nodal shape function gradients at evaluation points Xi.\n\n"
                "Args:\n"
                "    E (numpy.ndarray): `|# elem. nodes| x |# elements|` array of element node "
                "indices.\n"
                "    X (numpy.ndarray): `|# dims| x |# element vertices|` array of element "
                "vertices.\n"
                "    Xi (numpy.ndarray): `|# dims| x |# eval. pts.|` array of evaluation points in "
                "reference space.\n"
                "    element (EElement): Type of the finite element.\n"
                "    order (int): Order of the finite element.\n\n"
                "    dims (int): Number of spatial dimensions.\n"
                "Returns:\n"
                "    numpy.ndarray: `|# elem. nodes| x |# dims * # quad.pts.|` "
                "matrix of shape function gradients at mesh element quadrature points");
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat
