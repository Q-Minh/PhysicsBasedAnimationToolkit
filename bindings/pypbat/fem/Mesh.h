#ifndef PYPBAT_FEM_MESH_H
#define PYPBAT_FEM_MESH_H

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

namespace pbat {
namespace py {
namespace fem {

enum class EElement { Line, Triangle, Quadrilateral, Tetrahedron, Hexahedron };

class Mesh
{
  public:
    Mesh(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        EElement element,
        int order,
        int dims);

    template <class Func>
    void Apply(Func&& f) const;

    template <auto MaxQuadratureOrder, class Func>
    void ApplyWithQuadrature(Func&& f, int qOrder) const;

    MatrixX QuadraturePoints(int qOrder) const;
    VectorX QuadratureWeights(int qOrder) const;

    MatrixX const& X() const;
    IndexMatrixX const& E() const;

    MatrixX& X();
    IndexMatrixX& E();

    ~Mesh();

    EElement eElement;
    int kOrder;
    int kDims;

  private:
    void* mMesh;
};

void BindMesh(pybind11::module& m);

template <class Func>
inline void Mesh::Apply(Func&& f) const

{
    using namespace pbat::fem;
    using namespace pbat::common;
    ForValues<1, 2, 3>([&]<auto Order>() {
        ForTypes<
            Line<Order>,
            Triangle<Order>,
            Quadrilateral<Order>,
            Tetrahedron<Order>,
            Hexahedron<Order>>([&]<CElement ElementType>() {
            EElement constexpr eElementCandidate = []() {
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

            auto constexpr DimsIn = ElementType::kDims;
            ForRange<DimsIn, 3 + 1>([&]<auto Dims>() {
                if ((kOrder == Order) and (kDims == Dims) and (eElement == eElementCandidate))
                {
                    using MeshType = pbat::fem::Mesh<ElementType, Dims>;
                    MeshType* mesh = reinterpret_cast<MeshType*>(mMesh);
                    f(mesh);
                }
            });
        });
    });
}

template <auto MaxQuadratureOrder, class Func>
inline void Mesh::ApplyWithQuadrature(Func&& f, int qOrder) const
{
    if (qOrder > kMaxQuadratureOrder or qOrder < 1)
    {
        std::string const what = fmt::format(
            "Invalid quadrature order={}, supported orders are [1,{}]",
            qOrder,
            kMaxQuadratureOrder);
        throw std::invalid_argument(what);
    }
    Apply([&]<class MeshType>(MeshType* mesh) {
        pbat::common::ForRange<1, MaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {
            if (qOrder == QuadratureOrder)
            {
                f.template operator()<MeshType, QuadratureOrder>(mesh);
            }
        });
    });
}

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_MESH_H
