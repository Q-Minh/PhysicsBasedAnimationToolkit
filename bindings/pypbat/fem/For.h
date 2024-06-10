#ifndef PYPBAT_FEM_FOR_H
#define PYPBAT_FEM_FOR_H

#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/Hexahedron.h>
#include <pbat/fem/Line.h>
#include <pbat/fem/Mesh.h>
#include <pbat/fem/Quadrilateral.h>
#include <pbat/fem/Tetrahedron.h>
#include <pbat/fem/Triangle.h>

namespace pbat {
namespace py {
namespace fem {

template <class F>
constexpr void ForElementTypes(F&& f)
{
    auto constexpr kOrderMax = 3;
    pbat::common::ForRange<1, kOrderMax + 1>([&]<auto Order>() {
        pbat::common::ForTypes<
            pbat::fem::Line<Order>,
            pbat::fem::Triangle<Order>,
            pbat::fem::Quadrilateral<Order>,
            pbat::fem::Tetrahedron<Order>,
            pbat::fem::Hexahedron<Order>>(
            [&]<class ElementType>() { f.template operator()<ElementType>(); });
    });
}

template <class F>
constexpr void ForMeshTypes(F&& f)
{
    auto constexpr kDimsMax = 3;
    ForElementTypes([&]<class ElementType>() {
        pbat::common::ForRange<ElementType::kDims, kDimsMax + 1>([&]<auto Dims>() {
            using MeshType = pbat::fem::Mesh<ElementType, Dims>;
            f.template operator()<MeshType>();
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_FOR_H