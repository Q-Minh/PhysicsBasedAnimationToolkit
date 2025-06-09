#ifndef PYPBAT_FEM_MESH_H
#define PYPBAT_FEM_MESH_H

#include <Eigen/Core>
#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/fem/Hexahedron.h>
#include <pbat/fem/Line.h>
#include <pbat/fem/Mesh.h>
#include <pbat/fem/Quadrilateral.h>
#include <pbat/fem/Tetrahedron.h>
#include <pbat/fem/Triangle.h>
#include <pybind11/pybind11.h>
#include <type_traits>

namespace pbat {
namespace py {
namespace fem {

enum class EElement { Line, Triangle, Quadrilateral, Tetrahedron, Hexahedron };

template <class Func>
inline void ApplyToElement(EElement eElement, int order, Func&& f)
{
    if (order < 1 or order > 3)
    {
        throw std::invalid_argument(
            fmt::format("Invalid mesh order, expected 1 <= order <= 3, but got {}", order));
    }
    using namespace pbat::fem;
    using namespace pbat::common;
    ForRange<1, 3 + 1>([&]<auto Order>() {
        ForTypes<
            Line<Order>,
            Triangle<Order>,
            Quadrilateral<Order>,
            Tetrahedron<Order>,
            Hexahedron<Order>>([&]<CElement ElementType>() {
            EElement const eElementCandidate = []() {
                if constexpr (std::is_same_v<ElementType, Line<Order>>)
                    return EElement::Line;
                if constexpr (std::is_same_v<ElementType, Triangle<Order>>)
                    return EElement::Triangle;
                if constexpr (std::is_same_v<ElementType, Quadrilateral<Order>>)
                    return EElement::Quadrilateral;
                if constexpr (std::is_same_v<ElementType, Tetrahedron<Order>>)
                    return EElement::Tetrahedron;
                if constexpr (std::is_same_v<ElementType, Hexahedron<Order>>)
                    return EElement::Hexahedron;
            }();
            if ((order == Order) and (eElement == eElementCandidate))
            {
                f.template operator()<ElementType>();
            }
        });
    });
}

template <auto MaxQuadratureOrder, class Func>
inline void ApplyToElementWithQuadrature(EElement eElement, int order, int qOrder, Func&& f)
{
    if (qOrder < 1 or qOrder > 3)
    {
        throw std::invalid_argument(
            fmt::format("Invalid quadrature order, expected 1 <= qOrder <= 3, but got {}", qOrder));
    }
    ApplyToElement(
        eElement,
        order,
        [f = std::forward<Func>(f), qOrder]<pbat::fem::CElement ElementType>() {
            pbat::common::ForRange<1, MaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {
                if (QuadratureOrder == qOrder)
                {
                    f.template operator()<ElementType, QuadratureOrder>();
                }
            });
        });
}

template <class Func>
inline void ApplyToElementInDims(EElement eElement, int order, int dims, Func&& f)
{
    if (dims < 1 or dims > 3)
    {
        throw std::invalid_argument(
            fmt::format("Invalid mesh dimensions, expected 1 <= dims <= 3, but got {}", dims));
    }
    ApplyToElement(
        eElement,
        order,
        [f = std::forward<Func>(f), dims]<pbat::fem::CElement ElementType>() {
            auto constexpr DimsIn = ElementType::kDims;
            pbat::common::ForRange<DimsIn, 3 + 1>([&]<auto Dims>() {
                if (dims == Dims)
                {
                    f.template operator()<ElementType, Dims>();
                }
            });
        });
}

template <class TScalar, class TIndex, class Func>
inline void ApplyToMesh(EElement meshElement, int meshOrder, int meshDims, Func&& f)
{
    ApplyToElementInDims(
        meshElement,
        meshOrder,
        meshDims,
        [f = std::forward<Func>(f)]<pbat::fem::CElement TElement, auto Dims>() {
            using MeshType = pbat::fem::Mesh<TElement, Dims, TScalar, TIndex>;
            f.template operator()<MeshType>();
        });
}

template <auto MaxQuadratureOrder, class Func>
inline void
ApplyToElementInDimsWithQuadrature(EElement eElement, int order, int dims, int qOrder, Func&& f)
{
    if (qOrder > MaxQuadratureOrder or qOrder < 1)
    {
        std::string const what = fmt::format(
            "Invalid quadrature order={}, supported orders are [1,{}]",
            qOrder,
            MaxQuadratureOrder);
        throw std::invalid_argument(what);
    }
    ApplyToElementInDims(eElement, order, dims, [&]<pbat::fem::CElement ElementType, int Dims>() {
        pbat::common::ForRange<1, MaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {
            if (QuadratureOrder == qOrder)
            {
                f.template operator()<ElementType, Dims, QuadratureOrder>();
            }
        });
    });
}

template <auto MaxQuadratureOrder, class TScalar, class TIndex, class Func>
inline void
ApplyToMeshWithQuadrature(EElement meshElement, int meshOrder, int meshDims, int qOrder, Func&& f)
{
    ApplyToElementInDimsWithQuadrature(
        meshElement,
        meshOrder,
        meshDims,
        qOrder,
        [f = std::forward<Func>(
             f)]<pbat::fem::CElement ElementType, auto Dims, auto QuadratureOrder>() {
            using MeshType = pbat::fem::Mesh<ElementType, Dims, TScalar, TIndex>;
            f.template operator()<MeshType, QuadratureOrder>();
        });
}

void BindMesh(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_MESH_H
