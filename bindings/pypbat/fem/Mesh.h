#ifndef PYPBAT_FEM_MESH_H
#define PYPBAT_FEM_MESH_H

#include <pbat/fem/Concepts.h>
#include <pbat/fem/Hexahedron.h>
#include <pbat/fem/Line.h>
#include <pbat/fem/Quadrilateral.h>
#include <pbat/fem/Tetrahedron.h>
#include <pbat/fem/Triangle.h>
#include <pybind11/pybind11.h>
#include <string>

namespace pbat {
namespace py {
namespace fem {

void BindMesh(pybind11::module& m);

template <pbat::fem::CElement ElementType>
std::string ElementTypeName()
{
    auto constexpr kOrder = ElementType::kOrder;
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
}

template <pbat::fem::CMesh MeshType>
std::string MeshTypeName()
{
    auto constexpr kOrder      = MeshType::kOrder;
    auto constexpr kDims       = MeshType::kDims;
    using ElementType          = typename MeshType::ElementType;
    std::string const typeName = "Mesh_" + ElementTypeName<ElementType>() + "_Order_" +
                                 std::to_string(kOrder) + "_Dims_" + std::to_string(kDims);
    return typeName;
}

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_MESH_H