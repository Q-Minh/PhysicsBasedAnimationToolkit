#ifndef PBA_FEM_HYPER_ELASTIC_POTENTIAL_H
#define PBA_FEM_HYPER_ELASTIC_POTENTIAL_H

#include "Concepts.h"
#include "DeformationGradient.h"

#include <Eigen/SVD>
#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/math/linalg/SparsityPattern.h>
#include <pbat/physics/HyperElasticity.h>
#include <pbat/profiling/Profiling.h>
#include <span>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
struct HyperElasticPotential
{
  public:
    using SelfType           = HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>;
    using MeshType           = TMesh;
    using ElementType        = typename TMesh::ElementType;
    using ElasticEnergyType  = THyperElasticEnergy;
    using QuadratureRuleType = typename ElementType::template QuadratureType<QuadratureOrder>;
    static_assert(
        MeshType::kDims == ElasticEnergyType::kDims,
        "Embedding dimensions of mesh must match dimensionality of hyper elastic energy.");

    static auto constexpr kDims           = THyperElasticEnergy::kDims;
    static int constexpr kOrder           = ElementType::kOrder - 1;
    static int constexpr kQuadratureOrder = QuadratureOrder;

    SelfType& operator=(SelfType const&) = delete;

    HyperElasticPotential(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        Scalar Y,
        Scalar nu);

    template <class TDerivedY, class TDerivednu>
    HyperElasticPotential(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        Eigen::DenseBase<TDerivedY> const& Y,
        Eigen::DenseBase<TDerivednu> const& nu);

    template <class TDerived>
    HyperElasticPotential(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        Eigen::MatrixBase<TDerived> const& x,
        Scalar Y,
        Scalar nu);

    template <class TDerivedx, class TDerivedY, class TDerivednu>
    HyperElasticPotential(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe,
        Eigen::MatrixBase<TDerivedx> const& x,
        Eigen::DenseBase<TDerivedY> const& Y,
        Eigen::DenseBase<TDerivednu> const& nu);

    void PrecomputeHessianSparsity();

    template <class TDerived>
    void ComputeElementElasticity(
        Eigen::MatrixBase<TDerived> const& x,
        bool bWithGradient     = true,
        bool bWithHessian      = true,
        bool bUseSpdProjection = true);

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

    /**
     * @brief Transforms this element-wise gradient representation into the global gradient.
     * @return
     */
    VectorX ToVector() const;

    /**
     * @brief Computes the elastic potential
     * @return
     */
    Scalar Eval() const;

    Index InputDimensions() const;
    Index OutputDimensions() const;

    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh
    Eigen::Ref<MatrixX const>
        GNe; ///< |ElementType::kNodes| x |MeshType::kDims * # element quadrature points *
             ///< #elements| element shape function gradients
    Eigen::Ref<MatrixX const> detJe; ///< |# element quadrature points| x |#elements| matrix of
                                     ///< jacobian determinants at element quadrature points

    MatrixX mue;     ///< |#quad.pts.|x|#elements| 1st Lame coefficient
    MatrixX lambdae; ///< |#quad.pts.|x|#elements| 2nd Lame coefficient
    MatrixX He;      ///< |(ElementType::kNodes*kDims)| x |#elements *
                     ///< (ElementType::kNodes*kDims)| element hessian matrices
    MatrixX Ge;      ///< |ElementType::kNodes*kDims| x |#elements| element gradient vectors
    VectorX Ue;      ///< |#elements| x 1 element elastic potentials
    math::linalg::SparsityPattern GH; ///< Directed adjacency graph of hessian
};

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
inline HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    Scalar Y,
    Scalar nu)
    : HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>(
          meshIn,
          detJe,
          GNe,
          MatrixX::Constant(QuadratureRuleType::kPoints, meshIn.E.cols(), Y),
          MatrixX::Constant(QuadratureRuleType::kPoints, meshIn.E.cols(), nu))
{
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
template <class TDerivedY, class TDerivednu>
inline HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    Eigen::DenseBase<TDerivedY> const& Y,
    Eigen::DenseBase<TDerivednu> const& nu)
    : mesh(meshIn), detJe(detJe), GNe(GNe), mue(), lambdae(), He(), Ge(), Ue(), GH()
{
    std::tie(mue, lambdae)            = physics::LameCoefficients(Y.reshaped(), nu.reshaped());
    auto const numberOfElements       = mesh.E.cols();
    auto constexpr kNodesPerElement   = ElementType::kNodes;
    auto constexpr kDofsPerElement    = kNodesPerElement * kDims;
    auto constexpr kQuadPtsPerElement = QuadratureRuleType::kPoints;
    mue.resize(kQuadPtsPerElement, numberOfElements);
    lambdae.resize(kQuadPtsPerElement, numberOfElements);
    Ue.setZero(numberOfElements);
    Ge.setZero(kDofsPerElement, numberOfElements);
    He.setZero(kDofsPerElement, kDofsPerElement * numberOfElements);
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
template <class TDerived>
inline HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    Eigen::MatrixBase<TDerived> const& x,
    Scalar Y,
    Scalar nu)
    : HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>(
          meshIn,
          detJe,
          GNe,
          x,
          MatrixX::Constant(QuadratureRuleType::kPoints, meshIn.E.cols(), Y),
          MatrixX::Constant(QuadratureRuleType::kPoints, meshIn.E.cols(), nu))
{
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
template <class TDerivedx, class TDerivedY, class TDerivednu>
inline HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::HyperElasticPotential(
    MeshType const& meshIn,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe,
    Eigen::MatrixBase<TDerivedx> const& x,
    Eigen::DenseBase<TDerivedY> const& Y,
    Eigen::DenseBase<TDerivednu> const& nu)
    : HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>(meshIn, detJe, GNe, Y, nu)
{
    ComputeElementElasticity(x);
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
template <class TDerived>
inline void
HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::ComputeElementElasticity(
    Eigen::MatrixBase<TDerived> const& x,
    bool bWithGradient,
    bool bWithHessian,
    bool bUseSpdProjection)
{
    PBAT_PROFILE_NAMED_SCOPE("fem.HyperElasticPotential.ComputeElementElasticity");
    // Check inputs
    CheckValidState();
    auto const numberOfElements = mesh.E.cols();
    auto const numberOfNodes    = mesh.X.cols();
    if (x.size() != numberOfNodes * kDims)
    {
        std::string const what = fmt::format(
            "Generalized coordinate vector must have dimensions |#nodes|*kDims={}, but got "
            "x.size()={}",
            numberOfNodes * kDims,
            x.size());
        throw std::invalid_argument(what);
    }

    Ue.setZero();
    if (bWithGradient)
        Ge.setZero();
    if (bWithHessian)
        He.setZero();

    ElasticEnergyType Psi{};

    // Compute element elastic energies and their derivatives
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto constexpr kDofsPerElement  = kNodesPerElement * kDims;
    auto const wg                   = common::ToEigen(QuadratureRuleType::weights);
    if (not bWithGradient and not bWithHessian)
    {
        tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.reshaped(kDims, numberOfNodes)(Eigen::all, nodes);
            for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
            {
                auto constexpr kStride = MeshType::kDims * QuadratureRuleType::kPoints;
                auto const gradPhi     = GNe.block<kNodesPerElement, MeshType::kDims>(
                    0,
                    e * kStride + g * MeshType::kDims);
                auto const F = xe * gradPhi;
                auto psiF    = Psi.eval(F.reshaped(), mue(g, e), lambdae(g, e));
                Ue(e) += (wg(g) * detJe(g, e)) * psiF;
            }
        });
    }
    else if (bWithGradient and not bWithHessian)
    {
        tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.reshaped(kDims, numberOfNodes)(Eigen::all, nodes);
            auto ge          = Ge.col(e);
            for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
            {
                auto constexpr kStride = MeshType::kDims * QuadratureRuleType::kPoints;
                auto const gradPhi     = GNe.block<kNodesPerElement, MeshType::kDims>(
                    0,
                    e * kStride + g * MeshType::kDims);
                auto const F          = xe * gradPhi;
                auto [psiF, gradPsiF] = Psi.evalWithGrad(F.reshaped(), mue(g, e), lambdae(g, e));
                Ue(e) += (wg(g) * detJe(g, e)) * psiF;
                ge +=
                    (wg(g) * detJe(g, e)) * GradientWrtDofs<ElementType, kDims>(gradPsiF, gradPhi);
            }
        });
    }
    else if (not bWithGradient and bWithHessian)
    {
        tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.reshaped(kDims, numberOfNodes)(Eigen::all, nodes);
            auto he          = He.block<kDofsPerElement, kDofsPerElement>(0, e * kDofsPerElement);
            for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
            {
                auto constexpr kStride = MeshType::kDims * QuadratureRuleType::kPoints;
                auto const gradPhi     = GNe.block<kNodesPerElement, MeshType::kDims>(
                    0,
                    e * kStride + g * MeshType::kDims);
                auto const F  = xe * gradPhi;
                auto psiF     = Psi.eval(F.reshaped(), mue(g, e), lambdae(g, e));
                auto hessPsiF = Psi.hessian(F.reshaped(), mue(g, e), lambdae(g, e));
                Ue(e) += (wg(g) * detJe(g, e)) * psiF;
                he += (wg(g) * detJe(g, e)) * HessianWrtDofs<ElementType, kDims>(hessPsiF, gradPhi);
            }
        });
    }
    else
    {
        tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.reshaped(kDims, numberOfNodes)(Eigen::all, nodes);
            auto ge          = Ge.col(e);
            auto he          = He.block<kDofsPerElement, kDofsPerElement>(0, e * kDofsPerElement);
            for (auto g = 0; g < QuadratureRuleType::kPoints; ++g)
            {
                auto constexpr kStride = MeshType::kDims * QuadratureRuleType::kPoints;
                auto const gradPhi     = GNe.block<kNodesPerElement, MeshType::kDims>(
                    0,
                    e * kStride + g * MeshType::kDims);
                auto const F = xe * gradPhi;
                auto [psiF, gradPsiF, hessPsiF] =
                    Psi.evalWithGradAndHessian(F.reshaped(), mue(g, e), lambdae(g, e));
                Ue(e) += (wg(g) * detJe(g, e)) * psiF;
                ge +=
                    (wg(g) * detJe(g, e)) * GradientWrtDofs<ElementType, kDims>(gradPsiF, gradPhi);
                he += (wg(g) * detJe(g, e)) * HessianWrtDofs<ElementType, kDims>(hessPsiF, gradPhi);
            }
        });
    }
    if (bWithHessian and bUseSpdProjection)
    {
        tbb::parallel_for(Index{0}, Index{numberOfElements}, [&](Index e) {
            auto he = He.block<kDofsPerElement, kDofsPerElement>(0, e * kDofsPerElement);
            Eigen::JacobiSVD<Matrix<kDofsPerElement, kDofsPerElement>> SVD{};
            SVD.compute(he, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Vector<kDofsPerElement> sigma = SVD.singularValues();
            for (auto s = sigma.size() - 1; s >= 0; --s)
            {
                if (sigma(s) >= 0.)
                    break;
                sigma(s) = -sigma(s);
            }
            he = SVD.matrixU() * sigma.asDiagonal() * SVD.matrixV().transpose();
        });
    }
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
template <class TDerivedIn, class TDerivedOut>
inline void HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBAT_PROFILE_NAMED_SCOPE("fem.HyperElasticPotential.Apply");
    auto const numberOfDofs = InputDimensions();
    if (x.rows() != numberOfDofs or y.rows() != numberOfDofs or x.cols() != y.cols())
    {
        std::string const what = fmt::format(
            "Expected inputs and outputs to have rows |#nodes*kDims|={} and same number of "
            "columns, but got dimensions "
            "x,y=({},{}), ({},{})",
            numberOfDofs,
            x.rows(),
            x.cols(),
            y.rows(),
            y.cols());
        throw std::invalid_argument(what);
    }

    auto constexpr kDofsPerElement = kDims * ElementType::kNodes;
    auto const numberOfElements    = mesh.E.cols();
    // NOTE: Outer loop could be parallelized over columns, and using graph coloring, inner loop
    // could also be parallelized, if it's worth it.
    for (auto c = 0; c < x.cols(); ++c)
    {
        for (auto e = 0; e < numberOfElements; ++e)
        {
            auto const nodes = mesh.E.col(e);
            auto const he    = He.block<kDofsPerElement, kDofsPerElement>(0, e * kDofsPerElement);
            auto const xe    = x.col(c).reshaped(kDims, x.size() / kDims)(Eigen::all, nodes);
            auto ye          = y.col(c).reshaped(kDims, y.size() / kDims)(Eigen::all, nodes);
            ye.reshaped() += he * xe.reshaped();
        }
    }
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
inline void
HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::PrecomputeHessianSparsity()
{
    PBAT_PROFILE_NAMED_SCOPE("fem.HyperElasticPotential.PrecomputeHessianSparsity");
    auto const numberOfElements = mesh.E.cols();
    auto const kNodesPerElement = ElementType::kNodes;
    auto const kDofsPerElement  = kNodesPerElement * kDims;
    std::vector<Index> nonZeroRowIndices{};
    std::vector<Index> nonZeroColIndices{};
    nonZeroRowIndices.reserve(
        static_cast<std::size_t>(kDofsPerElement * kDofsPerElement * numberOfElements));
    nonZeroColIndices.reserve(
        static_cast<std::size_t>(kDofsPerElement * kDofsPerElement * numberOfElements));
    // Insert non-zero indices in the storage order of our He matrix of element hessians
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        for (auto j = 0; j < kNodesPerElement; ++j)
        {
            for (auto dj = 0; dj < kDims; ++dj)
            {
                for (auto i = 0; i < kNodesPerElement; ++i)
                {
                    for (auto di = 0; di < kDims; ++di)
                    {
                        nonZeroRowIndices.push_back(kDims * nodes(i) + di);
                        nonZeroColIndices.push_back(kDims * nodes(j) + dj);
                    }
                }
            }
        }
    }
    GH.Compute(OutputDimensions(), InputDimensions(), nonZeroRowIndices, nonZeroColIndices);
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
inline CSCMatrix
HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::ToMatrix() const
{
    PBAT_PROFILE_NAMED_SCOPE("fem.HyperElasticPotential.ToMatrix");
    if (!GH.IsEmpty())
    {
        using SpanType = std::span<Scalar const>;
        using SizeType = typename SpanType::size_type;
        return GH.ToMatrix(SpanType(He.data(), static_cast<SizeType>(He.size())));
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
                                static_cast<SparseIndex>(kDims * nodes(i) + di),
                                static_cast<SparseIndex>(kDims * nodes(j) + dj),
                                he(kDims * i + di, kDims * j + dj)});
        }

        auto const n = InputDimensions();
        CSCMatrix H(n, n);
        H.setFromTriplets(triplets.begin(), triplets.end());
        return H;
    }
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
inline VectorX HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::ToVector() const
{
    PBAT_PROFILE_NAMED_SCOPE("fem.HyperElasticPotential.ToVector");
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto const numberOfElements     = mesh.E.cols();
    auto const numberOfNodes        = mesh.X.cols();
    auto const n                    = InputDimensions();
    VectorX g                       = VectorX::Zero(n);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        auto const ge    = Ge.col(e).reshaped(kDims, kNodesPerElement);
        auto gi          = g.reshaped(kDims, numberOfNodes)(Eigen::all, nodes);
        gi += ge;
    }
    return g;
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
inline Scalar HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::Eval() const
{
    PBAT_PROFILE_NAMED_SCOPE("fem.HyperElasticPotential.Eval");
    return Ue.sum();
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
inline Index
HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::InputDimensions() const
{
    auto const numberOfNodes = mesh.X.cols();
    auto const numberOfDofs  = numberOfNodes * kDims;
    return numberOfDofs;
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
inline Index
HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::OutputDimensions() const
{
    return InputDimensions();
}

template <CMesh TMesh, physics::CHyperElasticEnergy THyperElasticEnergy, int QuadratureOrder>
inline void
HyperElasticPotential<TMesh, THyperElasticEnergy, QuadratureOrder>::CheckValidState() const
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
    auto constexpr kQuadPtsPerElements = QuadratureRuleType::kPoints;
    bool const bLameCoefficientsHaveCorrectDimensions =
        (mue.rows() == kQuadPtsPerElements) and (mue.cols() == numberOfElements) and
        (lambdae.rows() == kQuadPtsPerElements) and (lambdae.cols() == numberOfElements);
    if (not bLameCoefficientsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected quadrature point lame coefficients with dimensions {0}x{1} and "
            "{0}x{1} for mue and lambdae, but got {2}x{3} and {4}x{5}",
            kQuadPtsPerElements,
            numberOfElements,
            mue.rows(),
            mue.cols(),
            lambdae.rows(),
            lambdae.cols());
        throw std::invalid_argument(what);
    }
}

} // namespace fem
} // namespace pbat

#endif // PBA_FEM_HYPER_ELASTIC_POTENTIAL_H