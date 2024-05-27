#ifndef PBA_CORE_FEM_DIVERGENCE_VECTOR_H
#define PBA_CORE_FEM_DIVERGENCE_VECTOR_H

#include "Concepts.h"
#include "Jacobian.h"
#include "ShapeFunctionGradients.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/common/Profiling.h"

#include <exception>
#include <format>
#include <tbb/parallel_for.h>

namespace pba {
namespace fem {

template <CMesh TMesh, int Dims>
struct DivergenceVector
{
  public:
    using SelfType              = DivergenceVector<TMesh, Dims>;
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    static int constexpr kDims  = Dims;
    static int constexpr kOrder = ElementType::kOrder - 1;

    template <int ShapeFunctionOrder>
    struct OrderSelector
    {
        static auto constexpr kOrder = ShapeFunctionOrder - 1;
    };

    template <>
    struct OrderSelector<1>
    {
        static auto constexpr kOrder = 1;
    };

    using QuadratureRuleType =
        ElementType::template QuadratureType<OrderSelector<ElementType::kOrder>::kOrder>;

    template <class TDerived>
    DivergenceVector(MeshType const& mesh, Eigen::DenseBase<TDerived> const& Fe);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Transforms this matrix-free mass matrix representation into sparse compressed format.
     * @return
     */
    VectorX ToVector() const;

    /**
     * @brief Computes the piecewise divergence operator representations
     */
    template <class TDerivedF>
    void ComputeElementDivergence(Eigen::DenseBase<TDerivedF> const& Fe);

    MeshType const& mesh; ///< The finite element mesh
    MatrixX GNe;   ///< |ElementType::kNodes|x|kDims * QuadratureRuleType::kPoints * #elements|
                   ///< matrix of element shape function gradients at quadrature points
    MatrixX detJe; ///< |# element quadrature points|x|#elements| matrix of jacobian determinants at
                   ///< element quadrature points
    /*
     * div(F) = div(\sum F_i \phi_i) = \sum div(F_i \phi_i) = \sum \sum F_id d(\phi_i) / d(X_d)
     */
    MatrixX divE; ///< |ElementType::kNodes|x|#elements| matrix of element divergences at nodes
};

template <CMesh TMesh, int Dims>
template <class TDerived>
inline DivergenceVector<TMesh, Dims>::DivergenceVector(
    MeshType const& meshIn,
    Eigen::DenseBase<TDerived> const& Fe)
    : mesh(meshIn), GNe(), detJe()
{
    PBA_PROFILE_NAMED_SCOPE("Construct fem::DivergenceVector");
    auto const numberOfNodes = mesh.X.cols();
    if (Fe.rows() != kDims)
    {
        std::string const what = std::format(
            "LoadVector<TMesh,{0}> discretizes a {0}-dimensional load, but received "
            "{1}-dimensional input load",
            kDims,
            Fe.rows());
        throw std::invalid_argument(what);
    }
    if (Fe.cols() != numberOfNodes)
    {
        std::string const what = std::format(
            "Input load vector must be discretized at mesh nodes, but size was {}",
            Fe.cols());
        throw std::invalid_argument(what);
    }
    GNe   = ShapeFunctionGradients<QuadratureRuleType::kOrder>(mesh);
    detJe = DeterminantOfJacobian<QuadratureRuleType::kOrder>(mesh);
    ComputeElementDivergence(Fe);
}

template <CMesh TMesh, int Dims>
template <class TDerivedF>
inline void
DivergenceVector<TMesh, Dims>::ComputeElementDivergence(Eigen::DenseBase<TDerivedF> const& Fe)
{
    PBA_PROFILE_SCOPE;
    auto const numberOfElements = mesh.E.cols();
    divE.setZero(ElementType::kNodes, numberOfElements);
    auto const wg = common::ToEigen(QuadratureRuleType::weights);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes    = mesh.E.col(e);
        auto const vertices = nodes(ElementType::Vertices);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            auto constexpr kStride          = MeshType::kDims * QuadratureRuleType::kPoints;
            auto constexpr kNodesPerElement = ElementType::kNodes;
            auto const gradPhi =
                GNe.block<kNodesPerElement, MeshType::kDims>(0, e * kStride + g * MeshType::kDims);
            auto const F = Fe(Eigen::all, nodes);
            // div(F) = \sum_i \sum_d F_id d(\phi_i) / d(X_d)
            divE.col(e) =
                (wg(g) * detJe(g, e)) * (F.array().transpose() * gradPhi.array()).rowwise().sum();
        }
    });
}

template <CMesh TMesh, int Dims>
inline VectorX DivergenceVector<TMesh, Dims>::ToVector() const
{
    PBA_PROFILE_SCOPE;
    auto const numberOfNodes    = mesh.X.cols();
    auto const numberOfElements = mesh.E.cols();
    VectorX div                 = VectorX::Zero(numberOfNodes);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        div(nodes) += divE.col(e);
    }
    return div;
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_DIVERGENCE_VECTOR_H
