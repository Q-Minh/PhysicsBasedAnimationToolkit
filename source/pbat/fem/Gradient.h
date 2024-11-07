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
 */
template <CMesh TMesh>
struct Gradient
{
  public:
    using SelfType              = Gradient<TMesh>;
    using MeshType              = TMesh;
    using ElementType           = typename TMesh::ElementType;
    static int constexpr kDims  = MeshType::kDims;
    static int constexpr kOrder = (ElementType::kOrder > 1) ? (ElementType::kOrder - 1) : 1;

    /**
     * @brief
     * @param mesh
     * @param eg |#quad.pts.|x1 array of element indices associated with quadrature points
     * @param GNeg |#element nodes|x|#dims * #quad.pts.| matrix of element shape function gradients
     * at quadrature points points
     */
    Gradient(
        MeshType const& mesh,
        Eigen::Ref<IndexVectorX const> eg,
        Eigen::Ref<MatrixX const> const& GNeg);

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
    Index OutputDimensions() const { return kDims * eg.size(); }

    void CheckValidState() const;

    MeshType const& mesh; ///< The finite element mesh

    Eigen::Ref<IndexVectorX const>
        eg; ///< |#quad.pts.|x1 array of element indices corresponding to quadrature points
    Eigen::Ref<MatrixX const>
        GNeg; ///< |#element nodes|x|#dims * #quad.pts. * #elements|
              ///< matrix of element shape function gradients at quadrature points
};

template <CMesh TMesh>
inline Gradient<TMesh>::Gradient(
    MeshType const& mesh,
    Eigen::Ref<IndexVectorX const> eg,
    Eigen::Ref<MatrixX const> const& GNeg)
    : mesh(mesh), eg(eg), GNeg(GNeg)
{
}

template <CMesh TMesh>
inline CSCMatrix Gradient<TMesh>::ToMatrix() const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.Gradient.ToMatrix");
    using SparseIndex = typename CSCMatrix::StorageIndex;
    using Triplet     = Eigen::Triplet<Scalar, SparseIndex>;

    std::vector<Triplet> triplets{};
    triplets.reserve(static_cast<std::size_t>(GNeg.size()));
    auto const numberOfQuadraturePoints = eg.size();
    auto constexpr kNodesPerElement     = ElementType::kNodes;
    for (auto g = 0; g < numberOfQuadraturePoints; ++g)
    {
        auto const e     = eg(g);
        auto const nodes = mesh.E.col(e);
        auto const Geg   = GNeg.block<kNodesPerElement, kDims>(0, g * kDims);
        for (auto d = 0; d < kDims; ++d)
        {
            for (auto j = 0; j < kNodesPerElement; ++j)
            {
                auto const ni = static_cast<SparseIndex>(d * numberOfQuadraturePoints + g);
                auto const nj = static_cast<SparseIndex>(nodes(j));
                triplets.push_back(Triplet(ni, nj, Geg(j, d)));
            }
        }
    }
    CSCMatrix G(OutputDimensions(), InputDimensions());
    G.setFromTriplets(triplets.begin(), triplets.end());
    return G;
}

template <CMesh TMesh>
inline void Gradient<TMesh>::CheckValidState() const
{
    auto const numberOfQuadraturePoints = eg.size();
    auto constexpr kExpectedGNegRows    = ElementType::kNodes;
    auto const expectedGNegCols         = kDims * numberOfQuadraturePoints;
    bool const bShapeFunctionGradientsHaveCorrectDimensions =
        (GNeg.rows() == kExpectedGNegRows) and (GNeg.cols() == expectedGNegCols);
    if (not bShapeFunctionGradientsHaveCorrectDimensions)
    {
        std::string const what = fmt::format(
            "Expected shape function gradients at element quadrature points of dimensions "
            "|#nodes-per-element|={} x |#mesh-dims * #quad.pts.|={} for polynomial but got {}x{} "
            "instead",
            kExpectedGNegRows,
            expectedGNegCols,
            GNeg.rows(),
            GNeg.cols());
        throw std::invalid_argument(what);
    }
}

template <CMesh TMesh>
template <class TDerivedIn, class TDerivedOut>
inline void Gradient<TMesh>::Apply(
    Eigen::MatrixBase<TDerivedIn> const& x,
    Eigen::DenseBase<TDerivedOut>& y) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.Gradient.Apply");
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
    auto constexpr kNodesPerElement     = ElementType::kNodes;
    auto const numberOfQuadraturePoints = eg.size();
    for (auto c = 0; c < x.cols(); ++c)
    {
        for (auto g = 0; g < numberOfQuadraturePoints; ++g)
        {
            auto const e     = eg(g);
            auto const nodes = mesh.E.col(e);
            auto const xe    = x.col(c)(nodes);
            auto const Geg   = GNeg.block<kNodesPerElement, kDims>(0, g * kDims);
            for (auto d = 0; d < kDims; ++d)
            {
                y(d * numberOfQuadraturePoints + g) +=
                    Geg(Eigen::placeholders::all, d).transpose() * xe;
            }
        }
    }
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_GRADIENT_H