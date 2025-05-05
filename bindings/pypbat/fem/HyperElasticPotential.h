#ifndef PYPBAT_FEM_HYPERELASTICPOTENTIAL_H
#define PYPBAT_FEM_HYPERELASTICPOTENTIAL_H

#include "Mesh.h"

#include <pbat/fem/HyperElasticPotential.h>
#include <pbat/physics/SaintVenantKirchhoffEnergy.h>
#include <pbat/physics/StableNeoHookeanEnergy.h>
#include <pybind11/pybind11.h>
#include <tuple>
#include <type_traits>

namespace pbat {
namespace py {
namespace fem {

enum class EHyperElasticEnergy { SaintVenantKirchhoff, StableNeoHookean };

template <class Func>
inline void HepApplyToMesh(int meshDims, int meshOrder, EElement meshElement, Func&& f)
{
    using namespace pbat::fem;
    using namespace pbat::common;
    ForValues<1, 2, 3>([&]<auto Order>() {
        ForTypes<Line<Order>, Triangle<Order>, Quadrilateral<Order>, Tetrahedron<Order>>(
            [&]<CElement ElementType>() {
                EElement const eElementCandidate = []() {
                    if constexpr (std::is_same_v<ElementType, Line<Order>>)
                        return EElement::Line;
                    if constexpr (std::is_same_v<ElementType, Triangle<Order>>)
                        return EElement::Triangle;
                    if constexpr (std::is_same_v<ElementType, Quadrilateral<Order>>)
                        return EElement::Quadrilateral;
                    if constexpr (std::is_same_v<ElementType, Tetrahedron<Order>>)
                        return EElement::Tetrahedron;
                }();

                auto constexpr DimsIn = ElementType::kDims;
                if ((meshOrder == Order) and (meshDims == DimsIn) and
                    (meshElement == eElementCandidate))
                {
                    using MeshType = pbat::fem::Mesh<ElementType, DimsIn>;
                    f.template operator()<MeshType>();
                }
            });
    });

    // Elasticity can't handle cubic (or higher) hexahedra, because elastic hessians are stored on
    // the stack, and 3rd (or higher) order hexahedra have too many DOFs for the stack to allocate.
    ForValues<1, 2>([&]<auto Order>() {
        ForTypes<Hexahedron<Order>>([&]<CElement ElementType>() {
            EElement const eElementCandidate = EElement::Hexahedron;

            auto constexpr DimsIn = ElementType::kDims;
            if ((meshOrder == Order) and (meshDims == DimsIn) and
                (meshElement == eElementCandidate))
            {
                using MeshType = pbat::fem::Mesh<ElementType, DimsIn>;
                f.template operator()<MeshType>();
            }
        });
    });
}

class HyperElasticPotential
{
  public:
    HyperElasticPotential(
        Mesh const& M,
        Eigen::Ref<IndexVectorX const> const& eg,
        Eigen::Ref<VectorX const> const& wg,
        Eigen::Ref<MatrixX const> const& GNeg,
        Eigen::Ref<MatrixX const> const& lameg,
        EHyperElasticEnergy eHyperElasticEnergy);

    HyperElasticPotential(
        void* impl,
        EElement meshElement,
        int meshOrder,
        int meshDims,
        EHyperElasticEnergy eHyperElasticEnergy);

    HyperElasticPotential(HyperElasticPotential&& other);
    HyperElasticPotential& operator=(HyperElasticPotential&& other);

    HyperElasticPotential(HyperElasticPotential const&)            = delete;
    HyperElasticPotential& operator=(HyperElasticPotential const&) = delete;

    template <class Func>
    void Apply(Func&& f) const;

    void PrecomputeHessianSparsity();

    void ComputeElementElasticity(
        Eigen::Ref<VectorX const> const& x,
        bool bWithGradient,
        bool bWithHessian,
        bool bWithSpdProjection);

    Scalar Eval() const;
    VectorX ToVector() const;
    CSCMatrix ToMatrix() const;
    std::tuple<Index, Index> Shape() const;

    MatrixX const& Hessians() const;
    MatrixX& Hessians();

    MatrixX const& Gradients() const;
    MatrixX& Gradients();

    VectorX const& Potentials() const;
    VectorX& Potentials();

    ~HyperElasticPotential();

    EElement eMeshElement;
    int mMeshDims;
    int mMeshOrder;
    EHyperElasticEnergy eHyperElasticEnergy;
    int mDims;

  private:
    void* mHyperElasticPotential;
    bool bOwning;
};

void BindHyperElasticPotential(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_HYPERELASTICPOTENTIAL_H
