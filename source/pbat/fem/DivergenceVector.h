#ifndef PBAT_FEM_DIVERGENCE_VECTOR_H
#define PBAT_FEM_DIVERGENCE_VECTOR_H

#include "Concepts.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

template <CMesh TMesh, int Dims, int QuadratureOrder>
struct DivergenceVector
{
  public:
    using SelfType                        = DivergenceVector<TMesh, Dims, QuadratureOrder>;
    using MeshType                        = TMesh;
    using ElementType                     = typename TMesh::ElementType;
    static int constexpr kDims            = Dims;
    static int constexpr kOrder           = ElementType::kOrder - 1;
    static int constexpr kQuadratureOrder = QuadratureOrder;

    using QuadratureRuleType = typename ElementType::template QuadratureType<QuadratureOrder>;

    template <class TDerived>
    DivergenceVector(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        Eigen::DenseBase<TDerived> const& Fe);

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

    void CheckValidState() const;

    MeshType const& mesh;            ///< The finite element mesh
    Eigen::Ref<MatrixX const> detJe; ///< |# element quadrature points|x|#elements| matrix of
                                     ///< jacobian determinants at element quadrature points
    Eigen::Ref<MatrixX const>
        GNe; ///< |ElementType::kNodes|x|kDims * QuadratureRuleType::kPoints * #elements|
             ///< matrix of element shape function gradients at quadrature points
    /*
     * div(F) = div(\sum F_i \phi_i) = \sum div(F_i \phi_i) = \sum \sum F_id d(\phi_i) / d(X_d)
     */
    MatrixX divE; ///< |ElementType::kNodes|x|#elements| matrix of element divergences at nodes
};

template <CMesh TMesh, int Dims, int QuadratureOrder>
template <class TDerived>
inline DivergenceVector<TMesh, Dims, QuadratureOrder>::DivergenceVector(
    MeshType const& meshIn,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    Eigen::DenseBase<TDerived> const& Fe)
    : mesh(meshIn), GNe(GNe), detJe(detJe)
{
    auto const numberOfNodes = mesh.X.cols();
    if (Fe.rows() != kDims)
    {
        std::string const what = fmt::format(
            "DivergenceVector<TMesh,{0}> discretizes a {0}-dimensional load, but received "
            "{1}-dimensional input load",
            kDims,
            Fe.rows());
        throw std::invalid_argument(what);
    }
    if (Fe.cols() != numberOfNodes)
    {
        std::string const what = fmt::format(
            "Input load vector must be discretized at mesh nodes, but size was {}",
            Fe.cols());
        throw std::invalid_argument(what);
    }
    ComputeElementDivergence(Fe);
}

template <CMesh TMesh, int Dims, int QuadratureOrder>
template <class TDerivedF>
inline void DivergenceVector<TMesh, Dims, QuadratureOrder>::ComputeElementDivergence(
    Eigen::DenseBase<TDerivedF> const& Fe)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.DivergenceVector.ComputeElementDivergence");
    CheckValidState();
    auto const numberOfElements = mesh.E.cols();
    divE.setZero(ElementType::kNodes, numberOfElements);
    auto const wg = common::ToEigen(QuadratureRuleType::weights);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes = mesh.E.col(e);
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

template <CMesh TMesh, int Dims, int QuadratureOrder>
inline VectorX DivergenceVector<TMesh, Dims, QuadratureOrder>::ToVector() const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.DivergenceVector.ToVector");
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

template <CMesh TMesh, int Dims, int QuadratureOrder>
inline void DivergenceVector<TMesh, Dims, QuadratureOrder>::CheckValidState() const
{
    auto const numberOfElements       = mesh.E.cols();
    auto constexpr kExpectedDetJeRows = QuadratureRuleType::kPoints;
    auto const expectedDetJeCols      = numberOfElements;
    bool const bDeterminantsHaveCorrectDimensions =
        (detJe.rows() == kExpectedDetJeRows) and (detJe.cols() == expectedDetJeCols);
    if (not bDeterminantsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected determinants at element quadrature points of dimensions #quad.pts.={} x "
            "#elements={} for polynomial "
            "quadrature order={}, but got {}x{} instead.",
            kExpectedDetJeRows,
            expectedDetJeCols,
            QuadratureOrder,
            detJe.rows(),
            detJe.cols());
        throw std::invalid_argument(what);
    }
    auto constexpr kExpectedGNeRows = ElementType::kNodes;
    auto const expectedGNeCols = MeshType::kDims * QuadratureRuleType::kPoints * numberOfElements;
    bool const bShapeFunctionGradientsHaveCorrectDimensions =
        (GNe.rows() == kExpectedGNeRows) and (GNe.cols() == expectedGNeCols);
    if (not bShapeFunctionGradientsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected shape function gradients at element quadrature points of dimensions "
            "|#nodes-per-element|={} x |#mesh-dims * #quad.pts. * #elemens|={} for polynomiail "
            "quadrature order={}, but got {}x{} instead",
            kExpectedGNeRows,
            expectedGNeCols,
            QuadratureOrder,
            GNe.rows(),
            GNe.cols());
        throw std::invalid_argument(what);
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_DIVERGENCE_VECTOR_H
