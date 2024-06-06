#include "Mesh.h"

#include "pbat/common/ConstexprFor.h"
#include "pbat/fem/Hexahedron.h"
#include "pbat/fem/Line.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/Quadrilateral.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/fem/Triangle.h"

#include <string>
#include <type_traits>

namespace pbat {
namespace py {
namespace fem {

namespace pyb = pybind11;

void bind_mesh(pyb::module& m)
{
    auto constexpr kOrder   = 1;
    using ElementType       = pbat::fem::Tetrahedron<kOrder>;
    auto constexpr kDimsMax = 3;

    pbat::common::ForRange<1, 4>([&]<auto Order>() {
        pbat::common::ForTypes<
            pbat::fem::Line<Order>,
            pbat::fem::Triangle<Order>,
            pbat::fem::Quadrilateral<Order>,
            pbat::fem::Tetrahedron<Order>,
            pbat::fem::Hexahedron<Order>>([&]<class ElementType>() {
            pbat::common::ForRange<ElementType::kDims, kDimsMax + 1>([&]<auto Dims>() {
                using MeshType = pbat::fem::Mesh<ElementType, Dims>;

                std::string const elementTypeName = []() {
                    if constexpr (std::is_same_v<ElementType, pbat::fem::Line<Order>>)
                    {
                        return "line";
                    }
                    if constexpr (std::is_same_v<ElementType, pbat::fem::Triangle<Order>>)
                    {
                        return "triangle";
                    }
                    if constexpr (std::is_same_v<ElementType, pbat::fem::Quadrilateral<Order>>)
                    {
                        return "quadrilateral";
                    }
                    if constexpr (std::is_same_v<ElementType, pbat::fem::Tetrahedron<Order>>)
                    {
                        return "tetrahedron";
                    }
                    if constexpr (std::is_same_v<ElementType, pbat::fem::Hexahedron<Order>>)
                    {
                        return "hexahedron";
                    }
                }();

                std::string const className = "Mesh_" + elementTypeName + "_Order_" +
                                              std::to_string(Order) + "_Dims_" +
                                              std::to_string(Dims);

                pyb::class_<MeshType>(m, className.data())
                    .def(pyb::init<>())
                    .def(pyb::init<
                         Eigen::Ref<MatrixX const> const&,
                         Eigen::Ref<IndexMatrixX const> const&>())
                    .def_property_readonly_static(
                        "dims",
                        [](pyb::object /*self*/) { return MeshType::kDims; })
                    .def_property_readonly_static(
                        "element_type",
                        [=](pyb::object /*self*/) { return elementTypeName; })
                    .def_readwrite("E", &MeshType::E)
                    .def_readwrite("X", &MeshType::X);
            });
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat