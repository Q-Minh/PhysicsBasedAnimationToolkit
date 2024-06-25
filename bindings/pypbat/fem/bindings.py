autogendir = "gen"


def mesh_types_of(max_order=3):
    elements = ["Line", "Triangle",
                "Quadrilateral", "Tetrahedron", "Hexahedron"]
    ldims = [1, 2, 2, 3, 3]
    udims = 3
    _mesh_types = []
    for e, element in enumerate(elements):
        for order in range(1, max_order+1):
            for dims in range(ldims[e], udims+1):
                mesh_type = f"""pbat::fem::Mesh<pbat::fem::{
                    element}<{order}>,{dims}>"""
                mesh_type_py = f"""Mesh_{element.lower()}_Order_{
                    order}_Dims_{dims}"""
                includes = f"""#include <pbat/fem/{
                    element}.h>\n#include <pbat/fem/Mesh.h>\n"""
                mesh_type_filename = f"M{element[:3].lower()}{order}_{dims}"
                _mesh_types.append(
                    (mesh_type, mesh_type_py, includes, mesh_type_filename))
    return _mesh_types


def bind_gradient(mesh_types: list, max_qorder=6):
    headers = []
    sources = []
    for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types:
        for qorder in range(1, max_qorder+1):
            header = f"{autogendir}/Grad{qorder}{mesh_type_filename}.h"
            with open(header, 'w', encoding="utf-8") as file:
                code = f"""
#ifndef PYPBAT_FEM_GRADIENT_{qorder}_{mesh_type_py}_H
#define PYPBAT_FEM_GRADIENT_{qorder}_{mesh_type_py}_H

#include <pybind11/pybind11.h>

namespace pbat {{
namespace py {{
namespace fem {{

void BindGradient_{qorder}_{mesh_type_py}(pybind11::module& m);

}} // namespace fem
}} // namespace py
}} // namespace pbat

#endif // PYPBAT_FEM_GRADIENT_{qorder}_{mesh_type_py}_H
                """
                file.write(code)

            source = f"{autogendir}/Grad{qorder}{mesh_type_filename}.cpp"
            with open(source, 'w', encoding="utf-8") as file:
                code = f"""
#include "Grad{qorder}{mesh_type_filename}.h"

{mesh_includes}
#include <pbat/fem/Gradient.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {{
namespace py {{
namespace fem {{

void BindGradient_{qorder}_{mesh_type_py}(pybind11::module& m)
{{
    namespace pyb = pybind11;
    auto constexpr QuadratureOrder = {qorder};
    using MeshType = {mesh_type};
    using GradientMatrixType    = pbat::fem::GalerkinGradient<MeshType, QuadratureOrder>;
    std::string const className = "GalerkinGradientMatrix_QuadratureOrder_{qorder}_{mesh_type_py}";
    pyb::class_<GradientMatrixType>(m, className.data())
        .def(
            pyb::init([](MeshType const& mesh,
                            Eigen::Ref<MatrixX const> const& detJe,
                            Eigen::Ref<MatrixX const> const& GNe) {{
                return GradientMatrixType(mesh, detJe, GNe);
            }}),
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("GNe"))
        .def_property_readonly_static(
            "dims",
            [](pyb::object /*self*/) {{ return GradientMatrixType::kDims; }})
        .def_property_readonly_static(
            "order",
            [](pyb::object /*self*/) {{ return GradientMatrixType::kOrder; }})
        .def_property_readonly_static(
            "quadrature_order",
            [](pyb::object /*self*/) {{ return GradientMatrixType::kQuadratureOrder; }})
        .def("rows", &GradientMatrixType::OutputDimensions)
        .def("cols", &GradientMatrixType::InputDimensions)
        .def("to_matrix", &GradientMatrixType::ToMatrix)
        .def_readonly("Ge", &GradientMatrixType::Ge);
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat"""
                file.write(code)

            headers.append(header)
            sources.append(source)

    header = """
#ifndef PYPBAT_FEM_GRADIENT_H
#define PYPBAT_FEM_GRADIENT_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindGradient(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_GRADIENT_H
"""

    includes = "\n".join([f"#include \"Grad{qorder}{mesh_type_filename}.h\"" for qorder in range(
        1, max_qorder+1) for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    bind_calls = "\n".join([f"BindGradient_{qorder}_{mesh_type_py}(m);" for qorder in range(
        1, max_qorder+1) for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    source = f"""
#include "Gradient.h"

{includes}

namespace pbat {{
namespace py {{
namespace fem {{

void BindGradient(pybind11::module& m)
{{
    {bind_calls}
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""

    with open(f"{autogendir}/Gradient.h", 'w', encoding="utf-8") as fheader:
        fheader.write(header)
    with open(f"{autogendir}/Gradient.cpp", 'w', encoding="utf-8") as fsource:
        fsource.write(source)

    headers.append(f"{autogendir}/Gradient.h")
    sources.append(f"{autogendir}/Gradient.cpp")
    return (headers, sources)


def bind_hyper_elastic_potential(mesh_types: list, max_qorder: int = 6):
    headers = []
    sources = []

    psi_types = [
        ("SaintVenantKirchhoffEnergy", "StVk", "StVk"), ("StableNeoHookeanEnergy", "StableNeoHookean", "SNH")]
    for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types:
        # Order 3 hexahedron element has too many DOFs to be stack allocated
        # by Eigen in the fem::HessianWrtDofs calls, so skip it.
        if mesh_type_py == "Mesh_hexahedron_Order_3_Dims_3":
            continue

        for (psi_type, psi_type_py, psi_type_short) in psi_types:
            for qorder in range(1, max_qorder+1):
                header = f"""{
                    autogendir}/HEP{psi_type_short}_{qorder}{mesh_type_filename}.h"""
                with open(header, 'w', encoding="utf-8") as file:
                    code = f"""
#ifndef PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_{psi_type_py}_{qorder}_{mesh_type_py}_H
#define PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_{psi_type_py}_{qorder}_{mesh_type_py}_H

#include <pybind11/pybind11.h>

namespace pbat {{
namespace py {{
namespace fem {{

void BindHyperElasticPotential_{psi_type_py}_{qorder}_{mesh_type_py}(pybind11::module& m);

}} // namespace fem
}} // namespace py
}} // namespace pbat

#endif // PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_{psi_type_py}_{qorder}_{mesh_type_py}_H
"""
                    file.write(code)

                source = f"""{
                    autogendir}/HEP{psi_type_short}_{qorder}{mesh_type_filename}.cpp"""
                with open(source, 'w', encoding="utf-8") as file:
                    code = f"""
#include "HEP{psi_type_short}_{qorder}{mesh_type_filename}.h"

{mesh_includes}
#include <pbat/fem/HyperElasticPotential.h>
#include <pbat/physics/{psi_type}.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {{
namespace py {{
namespace fem {{

void BindHyperElasticPotential_{psi_type_py}_{qorder}_{mesh_type_py}(pybind11::module& m)
{{
    namespace pyb = pybind11;
    auto constexpr QuadratureOrder = {qorder};
    using MeshType = {mesh_type};
    using ElementType          = typename MeshType::ElementType;
    using HyperElasticEnergy   = pbat::physics::{psi_type}<MeshType::kDims>;
    using ElasticPotentialType = pbat::fem::
        HyperElasticPotential<MeshType, HyperElasticEnergy, QuadratureOrder>;
    std::string const className =
        "HyperElasticPotential_{psi_type_py}_QuadratureOrder_{qorder}_Dims_" + std::to_string(MeshType::kDims) +
        "_{mesh_type_py}";
    pyb::class_<ElasticPotentialType>(m, className.data())
        .def(
            pyb::init([](MeshType const& mesh,
                            Eigen::Ref<MatrixX const> const& detJe,
                            Eigen::Ref<MatrixX const> const& GNe,
                            Eigen::Ref<VectorX const> const& Y,
                            Eigen::Ref<VectorX const> const& nu) {{
                return ElasticPotentialType(mesh, detJe, GNe, Y, nu);
            }}),
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("GNe"),
            pyb::arg("Y"),
            pyb::arg("nu"))
        .def(
            pyb::init([](MeshType const& mesh,
                            Eigen::Ref<MatrixX const> const& detJe,
                            Eigen::Ref<MatrixX const> const& GNe,
                            Eigen::Ref<VectorX const> const& x,
                            Eigen::Ref<VectorX const> const& Y,
                            Eigen::Ref<VectorX const> const& nu) {{
                return ElasticPotentialType(mesh, detJe, GNe, x, Y, nu);
            }}),
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("GNe"),
            pyb::arg("x"),
            pyb::arg("Y"),
            pyb::arg("nu"))
        .def_property_readonly_static(
            "dims",
            [](pyb::object /*self*/) {{ return ElasticPotentialType::kDims; }})
        .def_property_readonly_static(
            "order",
            [](pyb::object /*self*/) {{ return ElasticPotentialType::kOrder; }})
        .def_property_readonly_static(
            "quadrature_order",
            [](pyb::object /*self*/) {{
                return ElasticPotentialType::kQuadratureOrder;
            }})
        .def(
            "precompute_hessian_sparsity",
            &ElasticPotentialType::PrecomputeHessianSparsity)
        .def(
            "compute_element_elasticity",
            [](ElasticPotentialType& U,
               Eigen::Ref<VectorX const> const& x,
               bool bWithGradient,
               bool bWithHessian) {{ U.ComputeElementElasticity(x, bWithGradient, bWithHessian); }},
            pyb::arg("x"),
            pyb::arg("grad") = true,
            pyb::arg("hess") = true)
        .def("to_matrix", &ElasticPotentialType::ToMatrix)
        .def("to_vector", &ElasticPotentialType::ToVector)
        .def("eval", &ElasticPotentialType::Eval)
        .def("rows", &ElasticPotentialType::OutputDimensions)
        .def("cols", &ElasticPotentialType::InputDimensions)
        .def_readwrite("mue", &ElasticPotentialType::mue)
        .def_readwrite("lambdae", &ElasticPotentialType::lambdae)
        .def_readonly("He", &ElasticPotentialType::He)
        .def_readonly("Ge", &ElasticPotentialType::Ge)
        .def_readonly("Ue", &ElasticPotentialType::Ue);
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""
                    file.write(code)

                headers.append(header)
                sources.append(source)

    header = """
#ifndef PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_H
#define PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindHyperElasticPotential(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_H
"""

    includes = "\n".join([f"#include \"HEP{psi_type_short}_{qorder}{mesh_type_filename}.h\""
                          for (psi_type, psi_type_py, psi_type_short) in psi_types
                          for qorder in range(1, max_qorder+1)
                          for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types
                          if mesh_type_py != "Mesh_hexahedron_Order_3_Dims_3"])

    bind_calls = "\n".join([f"BindHyperElasticPotential_{psi_type_py}_{qorder}_{mesh_type_py}(m);"
                            for (psi_type, psi_type_py, psi_type_short) in psi_types
                            for qorder in range(1, max_qorder+1)
                            for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types
                            if mesh_type_py != "Mesh_hexahedron_Order_3_Dims_3"])

    source = f"""
#include "HyperElasticPotential.h"

{includes}

namespace pbat {{
namespace py {{
namespace fem {{

void BindHyperElasticPotential(pybind11::module& m)
{{
    {bind_calls}
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""

    with open(f"{autogendir}/HyperElasticPotential.h", 'w', encoding="utf-8") as fheader:
        fheader.write(header)
    with open(f"{autogendir}/HyperElasticPotential.cpp", 'w', encoding="utf-8") as fsource:
        fsource.write(source)

    headers.append(f"{autogendir}/HyperElasticPotential.h")
    sources.append(f"{autogendir}/HyperElasticPotential.cpp")
    return (headers, sources)


def bind_jacobian(mesh_types: list, max_qorder: int = 6):
    headers = []
    sources = []

    for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types:
        header = f"{autogendir}/Jac{mesh_type_filename}.h"
        with open(header, 'w', encoding="utf-8") as file:
            code = f"""
#ifndef PYPBAT_FEM_JACOBIAN_{mesh_type_py}_H
#define PYPBAT_FEM_JACOBIAN_{mesh_type_py}_H

#include <pybind11/pybind11.h>

namespace pbat {{
namespace py {{
namespace fem {{

void BindJacobian_{mesh_type_py}(pybind11::module& m);

}} // namespace fem
}} // namespace py
}} // namespace pbat

#endif // PYPBAT_FEM_JACOBIAN_{mesh_type_py}_H
"""
            file.write(code)

        source = f"{autogendir}/Jac{mesh_type_filename}.cpp"
        with open(source, 'w', encoding="utf-8") as file:
            code = f"""
#include "Jac{mesh_type_filename}.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/common/ConstexprFor.h>
{mesh_includes}
#include <pbat/fem/Jacobian.h>
#include <pybind11/eigen.h>
#include <string>

namespace pbat {{
namespace py {{
namespace fem {{

void BindJacobian_{mesh_type_py}(pybind11::module& m)
{{
    namespace pyb = pybind11;
    using MeshType = {mesh_type};
    auto constexpr kMaxQuadratureOrder = {max_qorder};
    auto const throw_bad_quad_order    = [&](int qorder) {{
        std::string const what = fmt::format(
            "Invalid quadrature order={{}}, supported orders are [1,{{}}]",
            qorder,
            kMaxQuadratureOrder);
        throw std::invalid_argument(what);
    }};
    std::string const jacobianDeterminantsName = "jacobian_determinants_{mesh_type_py}";
    m.def(
        jacobianDeterminantsName.data(),
        [&](MeshType const& mesh, int qorder) -> MatrixX {{
            MatrixX R;
            pbat::common::ForRange<1, kMaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {{
                if (qorder == QuadratureOrder)
                {{
                    R = pbat::fem::DeterminantOfJacobian<QuadratureOrder, MeshType>(mesh);
                }}
            }});
            if (R.size() == 0)
                throw_bad_quad_order(qorder);
            return R;
        }},
        pyb::arg("mesh"),
        pyb::arg("quadrature_order"));
        
        std::string const referencePositionsName = "reference_positions_{mesh_type_py}";
        m.def(
            referencePositionsName.data(),
            [&](MeshType const& mesh,
                Eigen::Ref<IndexVectorX const> const& E,
                Eigen::Ref<MatrixX const> const& X,
                int maxIterations,
                Scalar eps) -> MatrixX {{
                return pbat::fem::ReferencePositions<MeshType>(mesh, E, X, maxIterations, eps);
            }},
            pyb::arg("mesh"),
            pyb::arg("E"),
            pyb::arg("X"),
            pyb::arg("max_iterations"),
            pyb::arg("epsilon"));
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""
            file.write(code)

        headers.append(header)
        sources.append(source)

    header = """
#ifndef PYPBAT_FEM_JACOBIAN_H
#define PYPBAT_FEM_JACOBIAN_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindJacobian(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_JACOBIAN_H
"""

    includes = "\n".join([f"#include \"Jac{mesh_type_filename}.h\""
                          for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    bind_calls = "\n".join([f"BindJacobian_{mesh_type_py}(m);"
                            for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    source = f"""
#include "Jacobian.h"

{includes}

namespace pbat {{
namespace py {{
namespace fem {{

void BindJacobian(pybind11::module& m)
{{
    {bind_calls}
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""

    with open(f"{autogendir}/Jacobian.h", 'w', encoding="utf-8") as fheader:
        fheader.write(header)
    with open(f"{autogendir}/Jacobian.cpp", 'w', encoding="utf-8") as fsource:
        fsource.write(source)

    headers.append(f"{autogendir}/Jacobian.h")
    sources.append(f"{autogendir}/Jacobian.cpp")
    return (headers, sources)


def bind_laplacian_matrix(mesh_types: list, max_qorder: int = 6):
    headers = []
    sources = []

    for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types:
        for qorder in range(1, max_qorder+1):
            header = f"{autogendir}/LM{qorder}{mesh_type_filename}.h"
            with open(header, 'w', encoding="utf-8") as file:
                code = f"""
#ifndef PYPBAT_FEM_LAPLACIAN_MATRIX_{qorder}_{mesh_type_py}_H
#define PYPBAT_FEM_LAPLACIAN_MATRIX_{qorder}_{mesh_type_py}_H

#include <pybind11/pybind11.h>

namespace pbat {{
namespace py {{
namespace fem {{

void BindLaplacianMatrix_{qorder}_{mesh_type_py}(pybind11::module& m);

}} // namespace fem
}} // namespace py
}} // namespace pbat

#endif // PYPBAT_FEM_LAPLACIAN_MATRIX_{qorder}_{mesh_type_py}_H
"""
                file.write(code)

            source = f"{autogendir}/LM{qorder}{mesh_type_filename}.cpp"
            with open(source, 'w', encoding="utf-8") as file:
                code = f"""
#include "LM{qorder}{mesh_type_filename}.h"

{mesh_includes}
#include <pbat/fem/LaplacianMatrix.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {{
namespace py {{
namespace fem {{

void BindLaplacianMatrix_{qorder}_{mesh_type_py}(pybind11::module& m)
{{
    namespace pyb = pybind11;
    using MeshType = {mesh_type};
    auto constexpr QuadratureOrder = {qorder};
    using LaplacianMatrixType =
        pbat::fem::SymmetricLaplacianMatrix<MeshType, QuadratureOrder>;
    std::string const className = "SymmetricLaplacianMatrix_QuadratureOrder_{qorder}_{mesh_type_py}";
    pyb::class_<LaplacianMatrixType>(m, className.data())
        .def(
            pyb::init([](MeshType const& mesh,
                            Eigen::Ref<MatrixX const> const& detJe,
                            Eigen::Ref<MatrixX const> const& GNe) {{
                return LaplacianMatrixType(mesh, detJe, GNe);
            }}),
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("GNe"))
        .def_property_readonly_static(
            "order",
            [](pyb::object /*self*/) {{ return LaplacianMatrixType::kOrder; }})
        .def_property_readonly_static(
            "quadrature_order",
            [](pyb::object /*self*/) {{ return LaplacianMatrixType::kQuadratureOrder; }})
        .def("to_matrix", &LaplacianMatrixType::ToMatrix)
        .def("rows", &LaplacianMatrixType::OutputDimensions)
        .def("cols", &LaplacianMatrixType::InputDimensions)
        .def_readonly("deltae", &LaplacianMatrixType::deltaE)
        .def_readwrite("dims", &LaplacianMatrixType::dims);
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
    """
                file.write(code)

            headers.append(header)
            sources.append(source)

    header = """
#ifndef PYPBAT_FEM_LAPLACIAN_MATRIX_H
#define PYPBAT_FEM_LAPLACIAN_MATRIX_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindLaplacianMatrix(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_LAPLACIAN_MATRIX_H
"""

    includes = "\n".join([f"#include \"LM{qorder}{mesh_type_filename}.h\""
                          for qorder in range(1, max_qorder+1)
                          for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    bind_calls = "\n".join([f"BindLaplacianMatrix_{qorder}_{mesh_type_py}(m);"
                            for qorder in range(1, max_qorder+1)
                            for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    source = f"""
#include "LaplacianMatrix.h"

{includes}

namespace pbat {{
namespace py {{
namespace fem {{

void BindLaplacianMatrix(pybind11::module& m)
{{
    {bind_calls}
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""

    with open(f"{autogendir}/LaplacianMatrix.h", 'w', encoding="utf-8") as fheader:
        fheader.write(header)
    with open(f"{autogendir}/LaplacianMatrix.cpp", 'w', encoding="utf-8") as fsource:
        fsource.write(source)

    headers.append(f"{autogendir}/LaplacianMatrix.h")
    sources.append(f"{autogendir}/LaplacianMatrix.cpp")
    return (headers, sources)


def bind_load_vector(mesh_types: list, max_qorder: int = 3):
    headers = []
    sources = []

    for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types:
        for qorder in range(1, max_qorder+1):
            header = f"{autogendir}/LV{qorder}{mesh_type_filename}.h"
            with open(header, 'w', encoding="utf-8") as file:
                code = f"""
#ifndef PYPBAT_FEM_LOAD_VECTOR_{qorder}_{mesh_type_py}_H
#define PYPBAT_FEM_LOAD_VECTOR_{qorder}_{mesh_type_py}_H

#include <pybind11/pybind11.h>

namespace pbat {{
namespace py {{
namespace fem {{

void BindLoadVector_{qorder}_{mesh_type_py}(pybind11::module& m);

}} // namespace fem
}} // namespace py
}} // namespace pbat

#endif // PYPBAT_FEM_LOAD_VECTOR_{qorder}_{mesh_type_py}_H
"""
                file.write(code)

            source = f"{autogendir}/LV{qorder}{mesh_type_filename}.cpp"
            with open(source, 'w', encoding="utf-8") as file:
                code = f"""
#include "LV{qorder}{mesh_type_filename}.h"

{mesh_includes}
#include <pbat/fem/LoadVector.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {{
namespace py {{
namespace fem {{

void BindLoadVector_{qorder}_{mesh_type_py}(pybind11::module& m)
{{
    namespace pyb = pybind11;
    using MeshType = {mesh_type};
    auto constexpr QuadratureOrder = {qorder};
    using LoadVectorType = pbat::fem::LoadVector<MeshType, QuadratureOrder>;
    std::string const className =
        "LoadVector_QuadratureOrder_{qorder}_{mesh_type_py}";
    pyb::class_<LoadVectorType>(m, className.data())
        .def(
            pyb::init([](MeshType const& mesh,
                            Eigen::Ref<MatrixX const> const& detJe,
                            Eigen::Ref<MatrixX const> const& fe,
                            int dims) {{
                return LoadVectorType(mesh, detJe, fe, dims);
            }}),
            pyb::arg("mesh"),
            pyb::arg("detJe"),
            pyb::arg("fe"),
            pyb::arg("dims"))
        .def_property_readonly_static(
            "order",
            [](pyb::object /*self*/) {{ return LoadVectorType::kOrder; }})
        .def_property_readonly_static(
            "quadrature_order",
            [](pyb::object /*self*/) {{ return LoadVectorType::kQuadratureOrder; }})
        .def_readonly("fe", &LoadVectorType::fe)
        .def("to_vector", &LoadVectorType::ToVector)
        .def(
            "set_load",
            [](LoadVectorType& f, Eigen::Ref<VectorX const> const& fe) {{
                f.SetLoad(fe);
            }},
            pyb::arg("fe"))
        .def_readonly("N", &LoadVectorType::N)
        .def_readwrite("dims", &LoadVectorType::dims);
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""
                file.write(code)

            headers.append(header)
            sources.append(source)

    header = """
#ifndef PYPBAT_FEM_LOAD_VECTOR_H
#define PYPBAT_FEM_LOAD_VECTOR_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindLoadVector(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_LOAD_VECTOR_H
"""

    includes = "\n".join([f"#include \"LV{qorder}{mesh_type_filename}.h\""
                          for qorder in range(1, max_qorder+1)
                          for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    bind_calls = "\n".join([f"BindLoadVector_{qorder}_{mesh_type_py}(m);"
                            for qorder in range(1, max_qorder+1)
                            for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    source = f"""
#include "LoadVector.h"

{includes}

namespace pbat {{
namespace py {{
namespace fem {{

void BindLoadVector(pybind11::module& m)
{{
    {bind_calls}
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""

    with open(f"{autogendir}/LoadVector.h", 'w', encoding="utf-8") as fheader:
        fheader.write(header)
    with open(f"{autogendir}/LoadVector.cpp", 'w', encoding="utf-8") as fsource:
        fsource.write(source)

    headers.append(f"{autogendir}/LoadVector.h")
    sources.append(f"{autogendir}/LoadVector.cpp")
    return (headers, sources)


def bind_mass_matrix(mesh_types: list, max_qorder: int = 6):
    headers = []
    sources = []

    for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types:
        for qorder in range(1, max_qorder+1):
            header = f"{autogendir}/MM{qorder}{mesh_type_filename}.h"
            with open(header, 'w', encoding="utf-8") as file:
                code = f"""
#ifndef PYPBAT_FEM_MASS_MATRIX_{qorder}_{mesh_type_py}_H
#define PYPBAT_FEM_MASS_MATRIX_{qorder}_{mesh_type_py}_H

#include <pybind11/pybind11.h>

namespace pbat {{
namespace py {{
namespace fem {{

void BindMassMatrix_{qorder}_{mesh_type_py}(pybind11::module& m);

}} // namespace fem
}} // namespace py
}} // namespace pbat

#endif // PYPBAT_FEM_MASS_MATRIX_{qorder}_{mesh_type_py}_H
"""
                file.write(code)

            source = f"{autogendir}/MM{qorder}{mesh_type_filename}.cpp"
            with open(source, 'w', encoding="utf-8") as file:
                code = f"""
#include "MM{qorder}{mesh_type_filename}.h"

{mesh_includes}
#include <pbat/fem/MassMatrix.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {{
namespace py {{
namespace fem {{

void BindMassMatrix_{qorder}_{mesh_type_py}(pybind11::module& m)
{{
namespace pyb = pybind11;
using MeshType = {mesh_type};
auto constexpr kQuadratureOrder = {qorder};
using MassMatrixType = pbat::fem::MassMatrix<MeshType, kQuadratureOrder>;
std::string const className =
    "MassMatrix_QuadratureOrder_{qorder}_{mesh_type_py}";
pyb::class_<MassMatrixType>(m, className.data())
    .def(
        pyb::init([](MeshType const& mesh, Eigen::Ref<MatrixX const> const& detJe) {{
            return MassMatrixType(mesh, detJe);
        }}),
        pyb::arg("mesh"),
        pyb::arg("detJe"))
    .def(
        pyb::init([](MeshType const& mesh,
                        Eigen::Ref<MatrixX const> const& detJe,
                        Scalar rho) {{ return MassMatrixType(mesh, detJe, rho); }}),
        pyb::arg("mesh"),
        pyb::arg("detJe"),
        pyb::arg("rho"))
    .def(
        pyb::init(
            [](MeshType const& mesh,
                Eigen::Ref<MatrixX const> const& detJe,
                VectorX const& rhoe) {{ return MassMatrixType(mesh, detJe, rhoe); }}),
        pyb::arg("mesh"),
        pyb::arg("detJe"),
        pyb::arg("rhoe"))
    .def_property_readonly_static(
        "order",
        [](pyb::object /*self*/) {{ return MassMatrixType::kOrder; }})
    .def_property_readonly_static(
        "quadrature_order",
        [](pyb::object /*self*/) {{ return MassMatrixType::kQuadratureOrder; }})
    .def_readonly("Me", &MassMatrixType::Me)
    .def("rows", &MassMatrixType::OutputDimensions)
    .def("cols", &MassMatrixType::InputDimensions)
    .def("to_matrix", &MassMatrixType::ToMatrix)
    .def(
        "compute_element_mass_matrices",
        [](MassMatrixType& M, VectorX const& rhoe) {{
            M.ComputeElementMassMatrices(rhoe);
        }},
        pyb::arg("rhoe"))
    .def_readwrite("dims", &MassMatrixType::dims);
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""
                file.write(code)

            headers.append(header)
            sources.append(source)

    header = """
#ifndef PYPBAT_FEM_MASS_MATRIX_H
#define PYPBAT_FEM_MASS_MATRIX_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindMassMatrix(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_MASS_MATRIX_H
"""

    includes = "\n".join([f"#include \"MM{qorder}{mesh_type_filename}.h\""
                          for qorder in range(1, max_qorder+1)
                          for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    bind_calls = "\n".join([f"BindMassMatrix_{qorder}_{mesh_type_py}(m);"
                            for qorder in range(1, max_qorder+1)
                            for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    source = f"""
#include "MassMatrix.h"

{includes}

namespace pbat {{
namespace py {{
namespace fem {{

void BindMassMatrix(pybind11::module& m)
{{
    {bind_calls}
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""

    with open(f"{autogendir}/MassMatrix.h", 'w', encoding="utf-8") as fheader:
        fheader.write(header)
    with open(f"{autogendir}/MassMatrix.cpp", 'w', encoding="utf-8") as fsource:
        fsource.write(source)

    headers.append(f"{autogendir}/MassMatrix.h")
    sources.append(f"{autogendir}/MassMatrix.cpp")
    return (headers, sources)


def bind_mesh(mesh_types: list):
    headers = []
    sources = []

    for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types:
        header = f"{autogendir}/{mesh_type_filename}.h"
        with open(header, 'w', encoding="utf-8") as file:
            code = f"""
#ifndef PYPBAT_FEM_MESH_{mesh_type_py}_H
#define PYPBAT_FEM_MESH_{mesh_type_py}_H

#include <pybind11/pybind11.h>

namespace pbat {{
namespace py {{
namespace fem {{

void BindMesh_{mesh_type_py}(pybind11::module& m);

}} // namespace fem
}} // namespace py
}} // namespace pbat

#endif // PYPBAT_FEM_MESH_{mesh_type_py}_H
"""
            file.write(code)

        source = f"{autogendir}/{mesh_type_filename}.cpp"
        with open(source, 'w', encoding="utf-8") as file:
            code = f"""
#include "{mesh_type_filename}.h"

{mesh_includes}
#include <pbat/fem/Mesh.h>
#include <pybind11/eigen.h>
#include <string>

namespace pbat {{
namespace py {{
namespace fem {{

void BindMesh_{mesh_type_py}(pybind11::module& m)
{{
    namespace pyb = pybind11;
    using MeshType = {mesh_type};
    std::string const elementTypeName = "{str(mesh_type_py).split("_")[1].capitalize()}";
    std::string const className = "{mesh_type_py}";
    pyb::class_<MeshType>(m, className.data())
        .def(pyb::init<>())
        .def(
            pyb::
                init<Eigen::Ref<MatrixX const> const&, Eigen::Ref<IndexMatrixX const> const&>(),
            pyb::arg("V"),
            pyb::arg("C"))
        .def_property_readonly_static(
            "dims",
            [](pyb::object /*self*/) {{ return MeshType::kDims; }})
        .def_property_readonly_static(
            "order",
            [](pyb::object /*self*/) {{ return MeshType::kOrder; }})
        .def_property_readonly_static(
            "element",
            [=](pyb::object /*self*/) {{ return elementTypeName; }})
        .def_readwrite("E", &MeshType::E)
        .def_readwrite("X", &MeshType::X);
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""
            file.write(code)

        headers.append(header)
        sources.append(source)

    header = """
#ifndef PYPBAT_FEM_MESH_H
#define PYPBAT_FEM_MESH_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindMesh(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_MESH_H
"""

    includes = "\n".join([f"#include \"{mesh_type_filename}.h\""
                          for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    bind_calls = "\n".join([f"BindMesh_{mesh_type_py}(m);"
                            for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    source = f"""
#include "Mesh.h"

{includes}

namespace pbat {{
namespace py {{
namespace fem {{

void BindMesh(pybind11::module& m)
{{
    {bind_calls}
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""

    with open(f"{autogendir}/Mesh.h", 'w', encoding="utf-8") as fheader:
        fheader.write(header)
    with open(f"{autogendir}/Mesh.cpp", 'w', encoding="utf-8") as fsource:
        fsource.write(source)

    headers.append(f"{autogendir}/Mesh.h")
    sources.append(f"{autogendir}/Mesh.cpp")
    return (headers, sources)


def bind_shape_functions(mesh_types: list, max_qorder: int = 6):
    headers = []
    sources = []

    for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types:
        header = f"{autogendir}/SF{mesh_type_filename}.h"
        with open(header, 'w', encoding="utf-8") as file:
            code = f"""
#ifndef PYPBAT_FEM_SHAPE_FUNCTIONS_{mesh_type_py}_H
#define PYPBAT_FEM_SHAPE_FUNCTIONS_{mesh_type_py}_H

#include <pybind11/pybind11.h>

namespace pbat {{
namespace py {{
namespace fem {{

void BindShapeFunctions_{mesh_type_py}(pybind11::module& m);

}} // namespace fem
}} // namespace py
}} // namespace pbat

#endif // PYPBAT_FEM_SHAPE_FUNCTIONS_{mesh_type_py}_H
"""
            file.write(code)

        source = f"{autogendir}/SF{mesh_type_filename}.cpp"
        with open(source, 'w', encoding="utf-8") as file:
            code = f"""
#include "SF{mesh_type_filename}.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/common/ConstexprFor.h>
{mesh_includes}
#include <pbat/fem/ShapeFunctions.h>
#include <pybind11/eigen.h>
#include <string>

namespace pbat {{
namespace py {{
namespace fem {{

void BindShapeFunctions_{mesh_type_py}(pybind11::module& m)
{{
    namespace pyb = pybind11;
    using MeshType = {mesh_type};
    auto constexpr kMaxQuadratureOrder = {max_qorder};
    auto const throw_bad_quad_order    = [&](int qorder) {{
        std::string const what = fmt::format(
            "Invalid quadrature order={{}}, supported orders are [1,{{}}]",
            qorder,
            kMaxQuadratureOrder);
        throw std::invalid_argument(what);
    }};
    std::string const meshTypeName = "{mesh_type_py}";

    std::string const integratedShapeFunctionsName =
        "integrated_shape_functions_" + meshTypeName;
    m.def(
        integratedShapeFunctionsName.data(),
        [&](MeshType const& mesh,
            Eigen::Ref<MatrixX const> const& detJe,
            int qorder) -> MatrixX {{
            MatrixX R;
            pbat::common::ForRange<1, kMaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {{
                if (qorder == QuadratureOrder)
                {{
                    R = pbat::fem::IntegratedShapeFunctions<QuadratureOrder, MeshType>(
                        mesh,
                        detJe);
                }}
            }});
            if (R.size() == 0)
                throw_bad_quad_order(qorder);
            return R;
        }},
        pyb::arg("mesh"),
        pyb::arg("detJe"),
        pyb::arg("quadrature_order"));
    std::string const shapeFunctionGradientsName = "shape_function_gradients_" + meshTypeName;
    m.def(
        shapeFunctionGradientsName.data(),
        [&](MeshType const& mesh, int qorder) -> MatrixX {{
            MatrixX R;
            pbat::common::ForRange<1, kMaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {{
                if (qorder == QuadratureOrder)
                {{
                    R = pbat::fem::ShapeFunctionGradients<QuadratureOrder>(mesh);
                }}
            }});
            if (R.size() == 0)
                throw_bad_quad_order(qorder);
            return R;
        }},
        pyb::arg("mesh"),
        pyb::arg("quadrature_order"));
        
    std::string const shapeFunctionsAtName = "shape_functions_at_" + meshTypeName;
    m.def(
        shapeFunctionsAtName.data(),
        [&](MeshType const& mesh, Eigen::Ref<MatrixX const> const& Xi) -> MatrixX {{
            return pbat::fem::ShapeFunctionsAt<typename MeshType::ElementType>(Xi);
        }}, 
        pyb::arg("mesh"),
        pyb::arg("Xi"));
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""
            file.write(code)

        headers.append(header)
        sources.append(source)

    header = """
#ifndef PYPBAT_FEM_SHAPE_FUNCTIONS_H
#define PYPBAT_FEM_SHAPE_FUNCTIONS_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindShapeFunctions(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_SHAPE_FUNCTIONS_H
"""

    includes = "\n".join([f"#include \"SF{mesh_type_filename}.h\""
                          for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    bind_calls = "\n".join([f"BindShapeFunctions_{mesh_type_py}(m);"
                            for mesh_type, mesh_type_py, mesh_includes, mesh_type_filename in mesh_types])

    source = f"""
#include "ShapeFunctions.h"

{includes}

namespace pbat {{
namespace py {{
namespace fem {{

void BindShapeFunctions(pybind11::module& m)
{{
    {bind_calls}
}}

}} // namespace fem
}} // namespace py
}} // namespace pbat
"""

    with open(f"{autogendir}/ShapeFunctions.h", 'w', encoding="utf-8") as fheader:
        fheader.write(header)
    with open(f"{autogendir}/ShapeFunctions.cpp", 'w', encoding="utf-8") as fsource:
        fsource.write(source)

    headers.append(f"{autogendir}/ShapeFunctions.h")
    sources.append(f"{autogendir}/ShapeFunctions.cpp")
    return (headers, sources)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog="PBAT FEM Python Bindings Generator",
        description="""Generates Python binding code for pbat::fem""",
    )
    parser.add_argument(
        "-c",
        "--cmake",
        help="Prints generated files to CMake via stdout",
        type=bool,
        dest="cmake")
    parser.add_argument(
        "-t",
        "--type",
        help="The type whose instantiations should be generated",
        type=str,
        dest="type")
    args = parser.parse_args()

    import os
    if not os.path.exists("gen"):
        os.mkdir("gen")

    meshes = mesh_types_of(max_order=3)
    headers, sources = None, None
    if args.type == "Gradient":
        header, sources = bind_gradient(meshes, max_qorder=5)
    if args.type == "HyperElasticPotential":
        headers, sources = bind_hyper_elastic_potential(meshes, max_qorder=6)
    if args.type == "Jacobian":
        headers, sources = bind_jacobian(meshes, max_qorder=6)
    if args.type == "LaplacianMatrix":
        headers, sources = bind_laplacian_matrix(meshes, max_qorder=4)
    if args.type == "LoadVector":
        headers, sources = bind_load_vector(meshes, max_qorder=3)
    if args.type == "MassMatrix":
        headers, sources = bind_mass_matrix(meshes, max_qorder=6)
    if args.type == "Mesh":
        headers, sources = bind_mesh(meshes)
    if args.type == "ShapeFunctions":
        headers, sources = bind_shape_functions(meshes, max_qorder=6)

    # headers = "\n".join([
    #     "\n".join(gradient_headers),
    #     "\n".join(hyper_elastic_potential_headers),
    #     "\n".join(jacobian_headers),
    #     "\n".join(laplacian_matrix_headers),
    #     "\n".join(load_vector_headers),
    #     "\n".join(mass_matrix_headers),
    #     "\n".join(mesh_headers),
    #     "\n".join(shape_functions_headers)
    # ])
    if args.cmake:
        import sys
        cmake_target_sources = ";".join(sources)
        sys.stdout.write(cmake_target_sources)
