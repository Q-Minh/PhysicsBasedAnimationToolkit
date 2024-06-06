#include "Mesh.h"

#include "For.h"

#include <pbat/fem/Mesh.h>
#include <pybind11/eigen.h>
#include <string>
#include <type_traits>

namespace pbat {
namespace py {
namespace fem {

namespace pyb = pybind11;

void BindMesh(pyb::module& m)
{
    ForMeshTypes([&]<class MeshType>() {
        std::string const className       = MeshTypeName<MeshType>();
        std::string const elementTypeName = ElementTypeName<typename MeshType::ElementType>();
        pyb::class_<MeshType>(m, className.data())
            .def(pyb::init<>())
            .def(
                pyb::
                    init<Eigen::Ref<MatrixX const> const&, Eigen::Ref<IndexMatrixX const> const&>(),
                pyb::arg("V"),
                pyb::arg("C"))
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