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
inline void ApplyToMesh(int meshDims, int meshOrder, EElement meshElement, Func&& f)
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

            auto constexpr DimsIn = ElementType::kDims;
            ForRange<DimsIn, 3 + 1>([&]<auto Dims>() {
                if ((meshOrder == Order) and (meshDims == Dims) and
                    (meshElement == eElementCandidate))
                {
                    using MeshType = pbat::fem::Mesh<ElementType, Dims>;
                    f.template operator()<MeshType>();
                }
            });
        });
    });
}

template <auto MaxQuadratureOrder, class Func>
inline void
ApplyToMeshWithQuadrature(int meshDims, int meshOrder, EElement meshElement, int qOrder, Func&& f)
{
    if (qOrder > MaxQuadratureOrder or qOrder < 1)
    {
        std::string const what = fmt::format(
            "Invalid quadrature order={}, supported orders are [1,{}]",
            qOrder,
            MaxQuadratureOrder);
        throw std::invalid_argument(what);
    }
    ApplyToMesh(meshDims, meshOrder, meshElement, [&]<class MeshType>() {
        pbat::common::ForRange<1, MaxQuadratureOrder + 1>([&]<auto QuadratureOrder>() {
            if (qOrder == QuadratureOrder)
            {
                f.template operator()<MeshType, QuadratureOrder>();
            }
        });
    });
}

class Mesh
{
  public:
    Mesh(
        Eigen::Ref<MatrixX const> const& V,
        Eigen::Ref<IndexMatrixX const> const& C,
        EElement element,
        int order,
        int dims);

    Mesh(void* meshImpl, EElement element, int order, int dims);

    template <class Func>
    void Apply(Func&& f) const;

    template <auto MaxQuadratureOrder, class Func>
    void ApplyWithQuadrature(Func&& f, int qOrder) const;

    template <class MeshType>
    MeshType* Raw();

    template <class MeshType>
    MeshType const* Raw() const;

    MatrixX QuadraturePoints(int qOrder) const;
    VectorX QuadratureWeights(int qOrder) const;

    Eigen::Map<MatrixX> X() const;
    Eigen::Map<IndexMatrixX> E() const;

    [[maybe_unused]] void* Impl() const { return mMesh; }
    [[maybe_unused]] void* Impl() { return mMesh; }

    ~Mesh();

    EElement eElement;
    int mOrder;
    int mDims;

  private:
    void* mMesh;
    bool bOwnMesh;
};

void BindMesh(pybind11::module& m);

template <class Func>
inline void Mesh::Apply(Func&& f) const
{
    ApplyToMesh(mDims, mOrder, eElement, [&]<class MeshType>() {
        MeshType* mesh = reinterpret_cast<MeshType*>(mMesh);
        f.template operator()<MeshType>(mesh);
    });
}

template <auto MaxQuadratureOrder, class Func>
inline void Mesh::ApplyWithQuadrature(Func&& f, int qOrder) const
{
    ApplyToMeshWithQuadrature<MaxQuadratureOrder>(
        mDims,
        mOrder,
        eElement,
        qOrder,
        [&]<class MeshType, auto QuadratureOrder>() {
            MeshType* mesh = reinterpret_cast<MeshType*>(mMesh);
            f.template operator()<MeshType, QuadratureOrder>(mesh);
        });
}

template <class MeshType>
inline MeshType* Mesh::Raw()
{
    MeshType* raw{nullptr};
    this->Apply([&]<class OtherMeshType>(OtherMeshType* mesh) {
        if constexpr (std::is_same_v<MeshType, OtherMeshType>)
            raw = mesh;
    });
    return raw;
}

template <class MeshType>
inline MeshType const* Mesh::Raw() const
{
    MeshType const* raw{nullptr};
    this->Apply([&]<class OtherMeshType>(OtherMeshType* mesh) {
        if constexpr (std::is_same_v<MeshType, OtherMeshType>)
            raw = mesh;
    });
    return raw;
}

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_MESH_H
