#ifndef PBAT_FEM_GRADIENT_H
#define PBAT_FEM_GRADIENT_H

#include "Concepts.h"
#include "ShapeFunctions.h"

#include <exception>
#include <fmt/core.h>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/profiling/Profiling.h>
#include <tbb/parallel_for.h>

namespace pbat {
namespace fem {

/**
 * @brief Represents an FEM gradient operator that takes some function discretized at mesh nodes,
 * and returns its gradient at element quadrature points:
 *
 * \sum_j u_j \nabla \phi_j(X_g) ] \forall e \in E
 *
 * @tparam TMesh
 * @tparam QuadratureOrder
 */
template <CMesh TMesh, int QuadratureOrder>
struct Gradient
{
  public:
    using SelfType              = Gradient<TMesh, QuadratureOrder>;
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    using QuadratureRuleType    = typename ElementType::template QuadratureType<QuadratureOrder>;
    static int constexpr kDims  = MeshType::kDims;
    static int constexpr kOrder = (ElementType::kOrder > 1) ? (ElementType::kOrder - 1) : 1;
    static int constexpr kQuadratureOrder = QuadratureOrder;

    /**
     * @brief
     * @param mesh
     * @param GNe |#element nodes|x|#dims * #quad.pts. * #elements|
                    ///< matrix of element shape function gradients at quadrature points
     * points
     */
    Gradient(MeshType const& mesh, Eigen::Ref<MatrixX const> const& GNe);

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
    Index OutputDimensions() const { return kDims * mesh.E.cols() * QuadratureRuleType::kPoints; }

    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh
    Eigen::Ref<MatrixX const>
        GNe; ///< |#element nodes|x|#dims * #quad.pts. * #elements|
             ///< matrix of element shape function gradients at quadrature points
};

template <CMesh TMesh, int QuadratureOrder>
inline Gradient<TMesh, QuadratureOrder>::Gradient(
    MeshType const& mesh,
    Eigen::Ref<MatrixX const> const& GNe)
    : mesh(mesh), GNe(GNe)
{
}

template <CMesh TMesh, int QuadratureOrder>
inline CSCMatrix Gradient<TMesh, QuadratureOrder>::ToMatrix() const
{
    PBAT_PROFILE_NAMED_SCOPE("fem.Gradient.ToMatrix");
    using SparseIndex = typename CSCMatrix::StorageIndex;
    using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;

    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(GNe.size()));
    auto const numberOfElements                = mesh.E.cols();
    auto constexpr kNodesPerElement            = ElementType::kNodes;
    auto constexpr kQuadPts                    = QuadratureRuleType::kPoints;
    auto const numberOfElementQuadraturePoints = numberOfElements * kQuadPts;
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes = mesh.E.col(e);
        auto const Ge    = GNe.block<kNodesPerElement, kQuadPts * kDims>(0, e * kQuadPts * kDims);
        for (auto g = 0; g < kQuadPts; ++g)
        {
            for (auto d = 0; d < kDims; ++d)
            {
                for (auto j = 0; j < Ge.rows(); ++j)
                {
                    auto const ni = static_cast<SparseIndex>(
                        d * numberOfElementQuadraturePoints + e * kQuadPts + g);
                    auto const nj = static_cast<SparseIndex>(nodes(j));
                    triplets.push_back(Triplet(ni, nj, Ge(j, g * kDims + d)));
                }
            }
        }
    }
    CSCMatrix G(OutputDimensions(), InputDimensions());
    G.setFromTriplets(triplets.begin(), triplets.end());
    return G;
}

template <CMesh TMesh, int QuadratureOrder>
inline void Gradient<TMesh, QuadratureOrder>::CheckValidState() const
{
    auto constexpr kQuadPts         = QuadratureRuleType::kPoints;
    auto const numberOfElements     = mesh.E.cols();
    auto constexpr kExpectedGNeRows = ElementType::kNodes;
    auto const expectedGNeCols      = kDims * kQuadPts * numberOfElements;
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

template <CMesh TMesh, int QuadratureOrder>
template <class TDerivedIn, class TDerivedOut>
inline void Gradient<TMesh, QuadratureOrder>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBAT_PROFILE_NAMED_SCOPE("fem.Gradient.Apply");
    // Check inputs
    bool const bDimensionsMatch = (x.cols() == y.cols()) and (x.rows() == InputDimensions()) and
                                  (y.rows() == OutputDimensions());
    if (not bDimensionsMatch)
    {
        std::string const what = fmt::format(
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
    auto constexpr kQuadPts         = QuadratureRuleType::kPoints;
    auto const numberOfElements     = mesh.E.cols();
    for (auto c = 0; c < x.cols(); ++c)
    {
        for (auto e = 0; e < numberOfElements; ++e)
        {
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.col(c)(nodes);
            auto const Ge = GNe.block<kNodesPerElement, kDims * kQuadPts>(0, e * kQuadPts * kDims);
            for (auto d = 0; d < kDims; ++d)
            {
                auto ye =
                    y.col(c).segment(d * numberOfElements * kQuadPts + e * kQuadPts, kQuadPts);
                for (auto g = 0; g < kQuadPts; ++g)
                {
                    for (auto i = 0; i < nodes.size(); ++i)
                    {
                        ye(g) += Ge(i, d + g * kDims) * xe(i);
                    }
                }
            }
        }
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_GRADIENT_H