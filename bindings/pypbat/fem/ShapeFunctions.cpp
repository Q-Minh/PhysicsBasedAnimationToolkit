#include "ShapeFunctions.h"

#include "Mesh.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/ShapeFunctions.h>
#include <string>
#include <variant>

namespace pbat {
namespace py {
namespace fem {

void BindShapeFunctions(nanobind::module_& m)
{
    namespace nb = nanobind;

    enum class EDType { float32, float64 };

    auto const GetDTypeNameOrDefault = [](nanobind::object const& dtype,
                                          EDType const defaultDType = EDType::float64) {
        namespace nb            = nanobind;
        bool const bDTypeIsNone = dtype.is_none();
        if (not bDTypeIsNone)
        {
            bool const bCanQueryClassName = (nb::hasattr(dtype, "__class__")) and
                                            (nb::hasattr(dtype.attr("__class__"), "__name__"));
            if (not bCanQueryClassName)
            {
                throw std::runtime_error(
                    "dtype must be a numpy dtype scalar or object, but could not query class "
                    "name");
            }
        }
        if (bDTypeIsNone)
        {
            return defaultDType;
        }
        else
        {
            std::string const dtypeName =
                nb::cast<std::string>(dtype.attr("__class__").attr("__name__"));
            if (dtypeName == "float32" or dtypeName == "Float32DType")
            {
                return EDType::float32;
            }
            else if (dtypeName == "float64" or dtypeName == "Float64DType")
            {
                return EDType::float64;
            }
            else
            {
                throw std::runtime_error(
                    "dtype must be a numpy dtype scalar or object, but got " + dtypeName);
            }
        }
    };

    using TScalar = pbat::Scalar;
    using TIndex  = pbat::Index;

    m.def(
        "element_shape_functions",
        [GetDTypeNameOrDefault](
            EElement eElement,
            int order,
            int quadratureOrder,
            nb::object dtype) {
            using FloatMatrixType  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
            using DoubleMatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
            using ReturnType       = std::variant<FloatMatrixType, DoubleMatrixType>;
            ReturnType N;
            auto constexpr kMaxQuadratureOrder =
                4; // For now, only support up to 4th order quadrature rules
            EDType const eDType = GetDTypeNameOrDefault(dtype);
            if (eDType == EDType::float32)
            {
                ApplyToElementWithQuadrature<kMaxQuadratureOrder>(
                    eElement,
                    order,
                    quadratureOrder,
                    [&]<class ElementType, auto QuadratureOrder>() {
                        N = FloatMatrixType(
                            pbat::fem::
                                ElementShapeFunctions<ElementType, QuadratureOrder, float>());
                    });
            }
            else if (eDType == EDType::float64)
            {
                ApplyToElementWithQuadrature<kMaxQuadratureOrder>(
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
                throw std::runtime_error("Unsupported dtype for element shape functions");
            }
            return N;
        },
        nb::arg("element"),
        nb::arg("order")            = 1,
        nb::arg("quadrature_order") = 1,
        nb::arg("dtype")            = nb::none(),
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
        [GetDTypeNameOrDefault](
            std::int64_t nElements,
            EElement eElement,
            int order,
            int quadratureOrder,
            nb::object dtype) {
            using FloatMatrixType  = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
            using DoubleMatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
            using ReturnType       = std::variant<FloatMatrixType, DoubleMatrixType>;
            ReturnType N;
            auto constexpr kMaxQuadratureOrder =
                4; // For now, only support up to 4th order quadrature rules
            EDType const eDType = GetDTypeNameOrDefault(dtype);
            if (eDType == EDType::float32)
            {
                ApplyToElementWithQuadrature<kMaxQuadratureOrder>(
                    eElement,
                    order,
                    quadratureOrder,
                    [&]<class ElementType, auto QuadratureOrder>() {
                        N = FloatMatrixType(
                            pbat::fem::ElementShapeFunctions<ElementType, QuadratureOrder, float>()
                                .replicate(1, nElements));
                    });
            }
            else if (eDType == EDType::float64)
            {
                ApplyToElementWithQuadrature<kMaxQuadratureOrder>(
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
                throw std::runtime_error("Unsupported dtype for shape functions");
            }
            return N;
        },
        nb::arg("n_elements"),
        nb::arg("element"),
        nb::arg("order")            = 1,
        nb::arg("quadrature_order") = 1,
        nb::arg("dtype")            = nb::none(),
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

    m.def(
        "shape_function_matrix",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           TIndex nNodes,
           EElement eElement,
           int order,
           int qOrder) {
            auto constexpr kMaxQuadratureOrder =
                4; // For now, only support up to 4th order quadrature rules
            Eigen::SparseMatrix<TScalar, Eigen::RowMajor, TIndex> N;
            ApplyToElementWithQuadrature<kMaxQuadratureOrder>(
                eElement,
                order,
                qOrder,
                [&]<class ElementType, int QuadratureOrder>() {
                    N = pbat::fem::ShapeFunctionMatrix<ElementType, QuadratureOrder, TScalar>(
                        E.template topRows<ElementType::kNodes>(),
                        nNodes);
                });
            return N;
        },
        nb::arg("E"),
        nb::arg("n_nodes"),
        nb::arg("element"),
        nb::arg("order")            = 1,
        nb::arg("quadrature_order") = 1,
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

    m.def(
        "shape_function_matrix_at",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           TIndex nNodes,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
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
        nb::arg("E"),
        nb::arg("n_nodes"),
        nb::arg("Xi"),
        nb::arg("element"),
        nb::arg("order") = 1,
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

    m.def(
        "shape_functions_at",
        [](nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
           EElement eElement,
           int order) {
            Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> N;
            ApplyToElement(eElement, order, [&]<class ElementType>() {
                N = pbat::fem::ShapeFunctionsAt<ElementType>(
                    Xi.template topRows<ElementType::kDims>());
            });
            return N;
        },
        nb::arg("Xi"),
        nb::arg("element"),
        nb::arg("order") = 1,
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

    m.def(
        "element_shape_function_gradients",
        [](nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
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
        nb::arg("Xi"),
        nb::arg("X"),
        nb::arg("element"),
        nb::arg("order") = 1,
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

    m.def(
        "shape_function_gradients",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
           EElement eElement,
           int order,
           int dims,
           int qOrder) {
            auto constexpr kMaxQuadratureOrder =
                4; // For now, only support up to 4th order quadrature rules
            Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> GNeg;
            ApplyToElementInDimsWithQuadrature<kMaxQuadratureOrder>(
                eElement,
                order,
                dims,
                qOrder,
                [&]<class ElementType, int Dims, int QuadratureOrder>() {
                    GNeg = pbat::fem::ShapeFunctionGradients<ElementType, Dims, QuadratureOrder>(
                        E.template topRows<ElementType::kNodes>(),
                        X.template topRows<Dims>());
                });
            return GNeg;
        },
        nb::arg("E"),
        nb::arg("X"),
        nb::arg("element"),
        nb::arg("order")            = 1,
        nb::arg("dims")             = 3,
        nb::arg("quadrature_order") = 1,
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

    m.def(
        "shape_function_gradients_at",
        [](nb::DRef<Eigen::Matrix<TIndex, Eigen::Dynamic, Eigen::Dynamic> const> E,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> X,
           nb::DRef<Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic> const> Xi,
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
        nb::arg("E"),
        nb::arg("X"),
        nb::arg("Xi"),
        nb::arg("element"),
        nb::arg("order") = 1,
        nb::arg("dims")  = 3,
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
}

} // namespace fem
} // namespace py
} // namespace pbat
