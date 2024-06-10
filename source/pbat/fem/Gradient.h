#ifndef PBAT_FEM_GRADIENT_H
#define PBAT_FEM_GRADIENT_H

#include "Concepts.h"
#include "ShapeFunctions.h"

#include <exception>
#include <format>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief Represents an FEM gradient operator that takes some function discretized at mesh nodes,
 * and returns its gradient in the Galerkin sense, i.e. the gradient at the i^{th} node is:
 *
 * \sum_j u_j [\sum_e \int_{\Omega^e} \nabla \phi_j(X) \phi_i(X) \partial \Omega^e]
 *
 * @tparam TMesh
 * @tparam QuadratureOrder
 */
template <CMesh TMesh, int QuadratureOrder>
struct GalerkinGradient
{
  public:
    using SelfType                        = GalerkinGradient<TMesh, QuadratureOrder>;
    using MeshType                        = TMesh;
    using ElementType                     = typename TMesh::ElementType;
    using QuadratureRuleType              = ElementType::template QuadratureType<QuadratureOrder>;
    static int constexpr kDims            = MeshType::kDims;
    static int constexpr kOrder           = 2 * ElementType::kOrder - 1;
    static int constexpr kQuadratureOrder = QuadratureOrder;

    /**
     * @brief
     * @param mesh
     * @param detJe |#quad.pts.|x|#elements| affine element jacobian determinants at quadrature
     * @param GNe |#element nodes|x|#dims * #quad.pts. * #elements|
                    ///< matrix of element shape function gradients at quadrature points
     * points
     */
    GalerkinGradient(
        MeshType const& mesh,
        Eigen::Ref<MatrixX const> const& detJe,
        Eigen::Ref<MatrixX const> const& GNe);

    SelfType& operator=(SelfType const&) = delete;

    /**
     * @brief Applies the gradient matrix as a linear operator on x, adding result to y.
     *
     * @tparam TDerivedIn
     * @tparam TDerivedOut
     * @param x
     * @param y
     */
    template <class TDerivedIn, class TDerivedOut>
    void Apply(Eigen::MatrixBase<TDerivedIn> const& x, Eigen::DenseBase<TDerivedOut>& y) const;

    /**
     * @brief Transforms this matrix-free mass matrix representation into sparse compressed format.
     * @return
     */
    CSCMatrix ToMatrix() const;

    Index InputDimensions() const { return mesh.X.cols(); }
    Index OutputDimensions() const { return kDims * InputDimensions(); }

    /**
     * @brief
     */
    void ComputeElementGalerkinGradientMatrices();

    void CheckValidState();

    MeshType const& mesh;            ///< The finite element mesh
    Eigen::Ref<MatrixX const> detJe; ///< |# element quadrature points| x |# elements| matrix of
                                     ///< jacobian determinants at element quadrature points
    Eigen::Ref<MatrixX const>
        GNe;    ///< |#element nodes|x|#dims * #quad.pts. * #elements|
                ///< matrix of element shape function gradients at quadrature points
    MatrixX Ge; ///< |#element nodes|x|#element nodes * #dims * #elements| matrix of element
                ///< Galerkin gradient matrices
};

template <CMesh TMesh, int QuadratureOrder>
inline GalerkinGradient<TMesh, QuadratureOrder>::GalerkinGradient(
    MeshType const& mesh,
    Eigen::Ref<MatrixX const> const& detJe,
    Eigen::Ref<MatrixX const> const& GNe)
    : mesh(mesh), detJe(detJe), GNe(GNe), Ge()
{
    ComputeElementGalerkinGradientMatrices();
}

template <CMesh TMesh, int QuadratureOrder>
inline CSCMatrix GalerkinGradient<TMesh, QuadratureOrder>::ToMatrix() const
{
    PBA_PROFILE_SCOPE;
    using SparseIndex = typename CSCMatrix::StorageIndex;
    using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;

    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(Ge.size()));
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto const numberOfElements     = mesh.E.cols();
    auto const numberOfNodes        = mesh.X.cols();
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        for (auto d = 0; d < kDims; ++d)
        {
            auto const Ged = Ge.block<kNodesPerElement, kNodesPerElement>(
                0,
                e * kDims * kNodesPerElement + d * kNodesPerElement);
            for (auto j = 0; j < Ged.cols(); ++j)
            {
                for (auto i = 0; i < Ged.rows(); ++i)
                {
                    auto const ni = static_cast<SparseIndex>(d * numberOfNodes + nodes(i));
                    auto const nj = static_cast<SparseIndex>(nodes(j));
                    triplets.push_back(Triplet(ni, nj, Ged(i, j)));
                }
            }
        }
    }
    CSCMatrix G(OutputDimensions(), InputDimensions());
    G.setFromTriplets(triplets.begin(), triplets.end());
    return G;
}

template <CMesh TMesh, int QuadratureOrder>
inline void GalerkinGradient<TMesh, QuadratureOrder>::ComputeElementGalerkinGradientMatrices()
{
    PBA_PROFILE_SCOPE;
    CheckValidState();
    // Expand into per-element matrices
    // [\sum_e \int_{\Omega^e} \nabla \phi_j(X) \phi_i(X) \partial \Omega^e]
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto const numberOfElements     = mesh.E.cols();
    Ge.setZero(kNodesPerElement, kNodesPerElement * kDims * numberOfElements);
    auto constexpr kQuadPts                    = QuadratureRuleType::kPoints;
    Matrix<kNodesPerElement, kQuadPts> const N = ShapeFunctions<ElementType, kQuadratureOrder>();
    auto const wg                              = common::ToEigen(QuadratureRuleType::weights);
    tbb::parallel_for(Index{0}, numberOfElements, [&](Index e) {
        for (auto d = 0; d < kDims; ++d)
        {
            auto Ged = Ge.block<kNodesPerElement, kNodesPerElement>(
                0,
                e * kDims * kNodesPerElement + d * kNodesPerElement);
            for (auto g = 0; g < kQuadPts; ++g)
            {
                auto const GPd = GNe.col(e * kDims * kQuadPts + g * kDims + d);
                auto const Ng  = N.col(g);
                Ged += (wg(g) * detJe(g, e)) * Ng * GPd.transpose();
            }
        }
    });
}

template <CMesh TMesh, int QuadratureOrder>
inline void GalerkinGradient<TMesh, QuadratureOrder>::CheckValidState()
{
    auto constexpr kQuadPts     = QuadratureRuleType::kPoints;
    auto const numberOfElements = mesh.E.cols();
    bool const bHasDeterminants = (detJe.rows() == kQuadPts) and (detJe.cols() == numberOfElements);
    if (not bHasDeterminants)
    {
        std::string const what = std::format(
            "Expected element jacobian determinants of dimensions {}x{} for element quadrature of "
            "order={}, but got {}x{}",
            kQuadPts,
            numberOfElements,
            kQuadratureOrder,
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
        std::string const what = std::format(
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

template <CMesh TMesh, int QuadratureOrder>
template <class TDerivedIn, class TDerivedOut>
inline void GalerkinGradient<TMesh, QuadratureOrder>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBA_PROFILE_SCOPE;
    // Check inputs
    bool const bDimensionsMatch = (x.cols() == y.cols()) and (x.rows() == InputDimensions()) and
                                  (y.rows() == OutputDimensions());
    if (not bDimensionsMatch)
    {
        std::string const what = std::format(
            "Expected input to have rows={} and output to have rows={}, and same number of "
            "columns, but got dimensions "
            "x,y=({},{}), ({},{})",
            InputDimensions(),
            OutputDimensions(),
            x.rows(),
            x.cols(),
            y.rows(),
            y.cols());
        throw std::invalid_argument(what);
    }
    // Compute gradient
    auto constexpr kNodesPerElement = ElementType::kNodes;
    auto const numberOfElements     = mesh.E.cols();
    for (auto c = 0; c < x.cols(); ++c)
    {
        for (auto e = 0; e < numberOfElements; ++e)
        {
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.col(c)(nodes);
            for (auto d = 0; d < kDims; ++d)
            {
                auto ye        = y.col(c).segment(d * numberOfElements, numberOfElements)(nodes);
                auto const Ged = Ge.block<kNodesPerElement, kNodesPerElement>(
                    0,
                    e * kDims * kNodesPerElement + d * kNodesPerElement);
                ye += Ged * xe;
            }
        }
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_GRADIENT_H