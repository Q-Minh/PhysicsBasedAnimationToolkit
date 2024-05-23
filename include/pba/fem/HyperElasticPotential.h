#ifndef PBA_FEM_HYPER_ELASTIC_POTENTIAL_H
#define PBA_FEM_HYPER_ELASTIC_POTENTIAL_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/fem/DeformationGradient.h"
#include "pba/physics/HyperElasticity.h"

#include <tbb/parallel_for.h>

namespace pba {
namespace fem {

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
struct HyperElasticPotential
{
  public:
    using SelfType          = HyperElasticPotential<TMesh, THyperElasticEnergy>;
    using MeshType          = TMesh;
    using ElementType       = typename TMesh::ElementType;
    using ElasticEnergyType = THyperElasticEnergy;

    static auto constexpr kDims = THyperElasticEnergy::kDims;
    static int constexpr kOrder = ElementType::kOrder - 1;

    template <int OrderPrivate>
    struct OrderSelector
    {
        static auto constexpr kOrder = OrderPrivate - 1;
    };

    template <>
    struct OrderSelector<1>
    {
        static auto constexpr kOrder = 1;
    };

    using QuadratureRuleType =
        ElementType::template QuadratureType<OrderSelector<ElementType::kOrder>::kOrder>;

    SelfType& operator=(SelfType const&) = delete;

    HyperElasticPotential(MeshType const& mesh);

    void ComputeElementElasticity();

    MeshType const& mesh; ///< The finite element mesh
    MatrixX He;           ///< |(ElementType::kNodes*kDims)| x |#elements *
                          ///< (ElementType::kNodes*kDims)| element hessian matrices
    MatrixX Ge;           ///< |ElementType::kNodes*kDims| x |#elements| element gradient vectors
    VectorX Ue;           ///< |#elements| x 1 element elastic potentials
};

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline HyperElasticPotential<TMesh, THyperElasticEnergy>::HyperElasticPotential(
    MeshType const& meshIn)
    : mesh(meshIn), He(), Ge(), Ue()
{
    ComputeElementElasticity();
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::ComputeElementElasticity()
{
    using AffineElementType         = typename ElementType::AffineBaseType;
    auto const numberOfElements     = mesh.E.cols();
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto constexpr kDofsPerElement  = kNodesPerElement * kDims;
    auto const Xg                   = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints);
    Ue.setZero(numberOfElements);
    Ge.setZero(kDofsPerElement, numberOfElements);
    He.setZero(kDofsPerElement, kDofsPerElement * numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                = mesh.E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = MeshType::kDims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = mesh.X(Eigen::all, vertices);
        auto ge                         = Ge.col(e);
        auto he = He.block(0, e * kDofsPerElement, kDofsPerElement, kDofsPerElement);
        Scalar detJ{};
        if constexpr (AffineElementType::bHasConstantJacobian)
            detJ = DeterminantOfJacobian(Jacobian<AffineElementType>({}, Ve));

        auto const wg = common::ToEigen(QuadratureRuleType::weights);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            auto const Xi = Xg.col(g);
            if constexpr (!AffineElementType::bHasConstantJacobian)
                detJ = DeterminantOfJacobian(
                    Jacobian<AffineElementType>(Xi.segment(1, ElementType::kDims), Ve));

            Matrix<kDims, kDims> const F = DeformationGradient<ElementType>(Xi, Ve);
            // auto const dFdx              = DeformationGradientDeriatives();
            auto const [psiF, gradPsiF, hessPsiF] = ElasticEnergyType{}.evalWithGradAndHessian(F);
            Ue(e) += wg(g) * psiF;
            // ge += wg(g) * dFdx.transpose() * gradPsiF;
            // he += (wg(g) * detJ) * (dFdx.transpose() * hessPsiF * dFdx);
        }
    });
}

} // namespace fem
} // namespace pba

#endif // PBA_FEM_HYPER_ELASTIC_POTENTIAL_H