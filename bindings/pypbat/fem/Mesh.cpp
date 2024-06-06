#include "Mesh.h"

#include "For.h"
#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/Hexahedron.h"
#include "pbat/fem/Line.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/Quadrilateral.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/fem/Triangle.h"

#include <pybind11/eigen.h>
#include <string>
#include <type_traits>

namespace pbat {
namespace py {
namespace fem {

namespace pyb = pybind11;

void bind_mesh(pyb::module& m)
{
    ForMeshTypes([&]<class MeshType>() {
        using ElementType                 = typename MeshType::ElementType;
        auto constexpr kOrder             = MeshType::kOrder;
        auto constexpr kDims              = MeshType::kDims;
        std::string const elementTypeName = []() {
            if constexpr (std::is_same_v<ElementType, pbat::fem::Line<kOrder>>)
            {
                return "line";
            }
            if constexpr (std::is_same_v<ElementType, pbat::fem::Triangle<kOrder>>)
            {
                return "triangle";
            }
            if constexpr (std::is_same_v<ElementType, pbat::fem::Quadrilateral<kOrder>>)
            {
                return "quadrilateral";
            }
            if constexpr (std::is_same_v<ElementType, pbat::fem::Tetrahedron<kOrder>>)
            {
                return "tetrahedron";
            }
            if constexpr (std::is_same_v<ElementType, pbat::fem::Hexahedron<kOrder>>)
            {
                return "hexahedron";
            }
        }();

        std::string const className = "Mesh_" + elementTypeName + "_Order_" +
                                      std::to_string(kOrder) + "_Dims_" + std::to_string(kDims);

        pyb::class_<MeshType>(m, className.data())
            .def(pyb::init<>())
            .def(
                pyb::
                    init<Eigen::Ref<MatrixX const> const&, Eigen::Ref<IndexMatrixX const> const&>())
            .def_property_readonly_static(
                "dims",
                [](pyb::object /*self*/) { return MeshType::kDims; })
            .def_property_readonly_static(
                "order",
                [](pyb::object /*self*/) { return MeshType::kOrder; })
            .def_property_readonly_static(
                "element_type",
                [=](pyb::object /*self*/) { return elementTypeName; })
            .def_readwrite("E", &MeshType::E)
            .def_readwrite("X", &MeshType::X);
    });
}

} // namespace fem
} // namespace py
} // namespace pbat