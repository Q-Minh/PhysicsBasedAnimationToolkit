#ifndef PBA_FEM_HYPER_ELASTIC_POTENTIAL_H
#define PBA_FEM_HYPER_ELASTIC_POTENTIAL_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/fem/DeformationGradient.h"
#include "pba/math/linalg/SparsityPattern.h"
#include "pba/physics/HyperElasticity.h"

#include <exception>
#include <format>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <tbb/parallel_for.h>
#include <utility>

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

    template <class TDerived>
    HyperElasticPotential(MeshType const& mesh, Eigen::MatrixBase<TDerived> const& x);

    void PrecomputeShapeFunctionGradients();
    void PrecomputeJacobianDeterminants();
    void PrecomputeHessianSparsity();

    template <class TDerived>
    void ComputeElementElasticity(Eigen::MatrixBase<TDerived> const& x);

    /**
     * @brief Applies the hessian matrix of this potential as a linear operator on x, adding result
     * to y.
     *
     * @tparam TDerivedIn
     * @tparam TDerivedOut
     * @param x
     * @param y
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Transforms this matrix-free hessian matrix representation into sparse compressed
     * format.
     * @return
     */
    CSCMatrix ToMatrix() const;

    Index InputDimensions() const;
    Index OutputDimensions() const;

    MeshType const& mesh; ///< The finite element mesh
    MatrixX He;           ///< |(ElementType::kNodes*kDims)| x |#elements *
                          ///< (ElementType::kNodes*kDims)| element hessian matrices
    MatrixX Ge;           ///< |ElementType::kNodes*kDims| x |#elements| element gradient vectors
    VectorX Ue;           ///< |#elements| x 1 element elastic potentials
    MatrixX GNe;   ///< |MeshType::kDims| x |ElementType::kNodes * # element quadrature points *
                   ///< #elements| element shape function gradients
    MatrixX detJe; ///< |# element quadrature points| x |#elements| matrix of jacobian determinants
                   ///< at element quadrature points
    math::linalg::SparsityPattern GH; ///< Directed adjacency graph of hessian
};

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerived>
inline HyperElasticPotential<TMesh, THyperElasticEnergy>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::MatrixBase<TDerived> const& x)
    : mesh(meshIn), He(), Ge(), Ue(), GH()
{
    PrecomputeShapeFunctionGradients();
    PrecomputeJacobianDeterminants();
    PrecomputeHessianSparsity();
    ComputeElementElasticity(x);
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerived>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::ComputeElementElasticity(
    Eigen::MatrixBase<TDerived> const& x)
{
    using AffineElementType     = typename ElementType::AffineBaseType;
    auto const numberOfElements = mesh.E.cols();
    auto const numberOfNodes    = mesh.X.cols();
    if (x.size() != numberOfNodes * kDims)
    {
        std::string const what = std::format(
            "Generalized coordinate vector must have dimensions |#nodes|*kDims={}, but got "
            "x.size()={}",
            numberOfNodes * kDims,
            x.size());
        throw std::invalid_argument(what);
    }
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto constexpr kDofsPerElement  = kNodesPerElement * kDims;
    Ue.setZero(numberOfElements);
    Ge.setZero(kDofsPerElement, numberOfElements);
    He.setZero(kDofsPerElement, kDofsPerElement * numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                      = mesh.E.col(e);
        auto const vertices                   = nodes(ElementType::Vertices);
        auto constexpr kMeshDims              = MeshType::kDims;
        auto constexpr kVertices              = AffineElementType::kNodes;
        Matrix<kMeshDims, kVertices> const Ve = mesh.X(Eigen::all, vertices);
        auto ge                               = Ge.col(e);
        auto he       = He.block<kDofsPerElement, kDofsPerElement>(0, e * kDofsPerElement);
        auto const xe = x.reshaped(kDims, numberOfNodes)(Eigen::all, nodes);
        auto const wg = common::ToEigen(QuadratureRuleType::weights);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            Matrix<MeshType::kDims, kNodesPerElement> const gradPhi =
                GNe.block<MeshType::kDims, kNodesPerElement>(
                    0,
                    e * kNodesPerElement + g * MeshType::kDims);
            Matrix<kDims, kDims> const F          = xe * gradPhi;
            auto const [psiF, gradPsiF, hessPsiF] = ElasticEnergyType{}.evalWithGradAndHessian(F);
            Ue(e) += (wg(g) * detJe(g, e)) * psiF;
            ge += (wg(g) * detJe(g, e)) * GradientWrtDofs<ElementType, kDims>(gradPsiF, gradPhi);
            he += (wg(g) * detJe(g, e)) * HessianWrtDofs<ElementType, kDims>(hessPsiF, gradPhi);
        }
    });
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
template <class TDerivedIn, class TDerivedOut>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    auto const numberOfNodes = mesh.X.cols();
    auto const numberOfDofs  = kDims * numberOfNodes;
    if (x.cols() != numberOfDofs and y.cols() != numberOfDofs)
    {
        std::string const what = std::format(
            "Expected inputs and outputs to have size |#nodes|*kDims={}, but got sizes x,y={},{}",
            numberOfDofs,
            x.cols(),
            y.cols());
        throw std::invalid_argument(what);
    }

    auto constexpr kDofsPerElement = kDims * ElementType::kNodes;
    auto const numberOfElements    = mesh.E.cols();
    // NOTE: Could be parallelized using graph coloring, if it's worth it.
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        auto const he    = He.block<kDofsPerElement, kDofsPerElement>(0, e * kDofsPerElement);
        auto const xe    = x.reshaped(kDims, x.size() / kDims)(Eigen::all, nodes);
        auto ye          = y.reshaped(kDims, y.size() / kDims)(Eigen::all, nodes);
        ye.reshaped() += he * xe.reshaped();
    }
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::PrecomputeShapeFunctionGradients()
{
    using AffineElementType         = typename ElementType::AffineBaseType;
    auto const numberOfElements     = mesh.E.cols();
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto const Xg                   = common::ToEigen(QuadratureRuleType::points)
                        .reshaped(QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints)
                        .bottomRows<ElementType::kDims>();
    GNe.resize(MeshType::kDims, kNodesPerElement * numberOfElements * QuadratureRuleType::kPoints);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                = mesh.E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = MeshType::kDims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = mesh.X(Eigen::all, vertices);
        for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
        {
            auto const gradPhi = BasisFunctionGradients<ElementType>(Xg.col(g), Ve);
            GNe.block<MeshType::kDims, kNodesPerElement>(
                0,
                e * kNodesPerElement + g * QuadratureRuleType::kPoints) = gradPhi;
        }
    });
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::PrecomputeJacobianDeterminants()
{
    using AffineElementType     = typename ElementType::AffineBaseType;
    auto const numberOfElements = mesh.E.cols();
    auto const Xg               = common::ToEigen(QuadratureRuleType::points)
                        .reshaped<QuadratureRuleType::kDims + 1, QuadratureRuleType::kPoints>()
                        .bottomRows<ElementType::kDims>();
    detJe.resize(QuadratureRuleType::kPoints, numberOfElements);
    tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
        auto const nodes                = mesh.E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = MeshType::kDims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = mesh.X(Eigen::all, vertices);
        if constexpr (AffineElementType::bHasConstantJacobian)
        {
            Scalar const detJ = DeterminantOfJacobian(Jacobian<AffineElementType>({}, Ve));
            detJe.col(e).setConstant(detJ);
        }
        else
        {
            auto const wg = common::ToEigen(QuadratureRuleType::weights);
            for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
            {
                Scalar const detJ =
                    DeterminantOfJacobian(Jacobian<AffineElementType>(Xg.col(g), Ve));
                detJe(g, e) = detJ;
            }
        }
    });
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy>::PrecomputeHessianSparsity()
{
    using IndexPair = std::pair<Index, Index>;
    std::vector<IndexPair> nonZeroIndices{};
    auto const numberOfElements = mesh.E.cols();
    auto const kNodesPerElement = ElementType::kNodes;
    auto const kDofsPerElement  = kNodesPerElement * kDims;
    nonZeroIndices.reserve(kDofsPerElement * kDofsPerElement * numberOfElements);
    // Insert non-zero indices in the storage order of our He matrix of element hessians
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        for (auto j = 0; j < kNodesPerElement; ++j)
            for (auto dj = 0; dj < kDims; ++dj)
                for (auto i = 0; i < kNodesPerElement; ++i)
                    for (auto di = 0; di < kDims; ++di)
                        nonZeroIndices.push_back({kDims * nodes(i) + di, kDims * nodes(j) + dj});
    }
    auto const i = std::views::elements<0>(nonZeroIndices);
    auto const j = std::views::elements<1>(nonZeroIndices);
    GH.Compute(OutputDimensions(), InputDimensions(), i, j);
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline CSCMatrix HyperElasticPotential<TMesh, THyperElasticEnergy>::ToMatrix() const
{
    if (!GH.IsEmpty())
    {
        return GH.ToMatrix(std::span<Scalar const>(He.data(), He.size()));
    }
    else
    {
        // Construct hessian from triplets
        using SparseIndex = typename CSCMatrix::StorageIndex;
        using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;
        std::vector<Triplet> triplets{};
        triplets.reserve(static_cast<std::size_t>(He.size()));
        auto const numberOfElements = mesh.E.cols();
        for (auto e = 0; e < numberOfElements; ++e)
        {
            auto const nodes     = mesh.E.col(e);
            auto constexpr Hrows = ElementType::kNodes * kDims;
            auto constexpr Hcols = Hrows;
            auto const he        = He.block<Hrows, Hcols>(0, e * Hcols);
            for (auto j = 0; j < ElementType::kNodes; ++j)
                for (auto dj = 0; dj < kDims; ++dj)
                    for (auto i = 0; i < ElementType::kNodes; ++i)
                        for (auto di = 0; di < kDims; ++di)
                            triplets.push_back(Triplet{
                                kDims * nodes(i) + di,
                                kDims * nodes(j) + dj,
                                he(kDims * i + di, kDims * j + dj)});
        }

        auto const n = InputDimensions();
        CSCMatrix H(n, n);
        H.setFromTriplets(triplets.begin(), triplets.end());
        return H;
    }
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline Index HyperElasticPotential<TMesh, THyperElasticEnergy>::InputDimensions() const
{
    auto const numberOfNodes = mesh.X.cols();
    auto const numberOfDofs  = numberOfNodes * kDims;
    return numberOfDofs;
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy>
inline Index HyperElasticPotential<TMesh, THyperElasticEnergy>::OutputDimensions() const
{
    return InputDimensions();
}

} // namespace fem
} // namespace pba

#endif // PBA_FEM_HYPER_ELASTIC_POTENTIAL_H