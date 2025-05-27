/**
 * @file Mesh.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Finite element mesh API and implementation.
 * @date 2025-02-11
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_FEM_MESH_H
#define PBAT_FEM_MESH_H

#include "Concepts.h"
#include "Jacobian.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/common/Eigen.h"
#include "pbat/math/Rational.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <exception>
#include <map>
#include <numeric>
#include <ranges>

namespace pbat {
namespace fem {

/**
 * @brief A generic stateful finite element mesh representation.
 *
 * @tparam TElement Type satisfying concept CElement
 * @tparam Dims Embedding dimensions of the mesh
 */
template <CElement TElement, int Dims>
struct Mesh
{
    using ElementType                     = TElement; ///< Underlying finite element type
    static int constexpr kDims            = Dims;     ///< Embedding dimensions of the mesh
    static int constexpr kOrder           = ElementType::kOrder; ///< Shape function order
    static int constexpr kNodesPerElement = ElementType::kNodes; ///< Number of nodes per element

    Mesh() = default;
    /**
     * @brief Constructs a finite element mesh given some input geometric mesh.
     *
     * @warning The cells of the input mesh should list its vertices in Lagrange order.
     * @param V `Dims x |# vertices|` matrix of vertex positions
     * @param C `|Element::AffineBase::Vertices| x |# cells|` matrix of cell vertex indices into V
     */
    Mesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);
    /**
     * @brief Constructs a finite element mesh given some input geometric mesh.
     *
     * @warning The cells of the input mesh should list its vertices in Lagrange order.
     * @param V `Dims x |# vertices|` matrix of vertex positions
     * @param C `|Element::AffineBase::Vertices| x |# cells|` matrix of cell vertex indices into V
     */
    void Construct(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);
    /**
     * @brief Compute quadrature points in domain space on this mesh.
     * @tparam QuadratureOrder Quadrature order
     * @return `kDims x |# element quad.pts.|` matrix of quadrature points
     */
    template <int QuadratureOrder>
    MatrixX QuadraturePoints() const;
    /**
     * @brief Obtain quadrature weights on the reference element of this mesh
     * @tparam QuadratureOrder Quadrature order
     * @return `|# element quad.pts.|` vector of quadrature weights
     */
    template <int QuadratureOrder>
    Vector<TElement::template QuadratureType<QuadratureOrder>::kPoints> QuadratureWeights() const;

    Matrix<kDims, Eigen::Dynamic> X; ///< `kDims x |# nodes|` nodal positions
    IndexMatrix<kNodesPerElement, Eigen::Dynamic>
        E; ///< `|Element::Nodes| x |# elements|` element nodal indices
};

/**
 * @brief A non-owning view over a linear finite element mesh.
 *
 * @tparam TElement Type satisfying concept CElement
 * @tparam Dims Embedding dimensions of the mesh
 * @pre TElement::kOrder == 1
 */
template <CElement TElement, int Dims>
struct LinearMeshView
{
    using ElementType                     = TElement; ///< Underlying finite element type
    static int constexpr kDims            = Dims;     ///< Embedding dimensions of the mesh
    static int constexpr kOrder           = ElementType::kOrder; ///< Shape function order
    static int constexpr kNodesPerElement = ElementType::kNodes; ///< Number of nodes per element

    /**
     * @brief Constructs a finite element mesh given some input geometric mesh.
     *
     * @warning The cells of the input mesh should list its vertices in Lagrange order.
     * @param V `kDims x |# vertices|` matrix of vertex positions
     * @param C `|Element::AffineBase::Vertices| x |# cells|` matrix of cell vertex indices into V
     */
    LinearMeshView(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);

    /**
     * @brief Compute quadrature points in domain space for on this mesh.
     * @tparam QuadratureOrder Quadrature order
     * @return `kDims x |# element quad.pts.|` matrix of quadrature points
     */
    template <int QuadratureOrder>
    MatrixX QuadraturePoints() const;
    /**
     * @brief Obtain quadrature weights on the reference element of this mesh
     * @tparam QuadratureOrder Quadrature order
     * @return `|# element quad.pts.|` vector of quadrature weights
     */
    template <int QuadratureOrder>
    Vector<TElement::template QuadratureType<QuadratureOrder>::kPoints> QuadratureWeights() const;

    Eigen::Ref<Matrix<kDims, Eigen::Dynamic> const> X; ///< `kDims x |# nodes|` nodal positions
    Eigen::Ref<IndexMatrix<kNodesPerElement, Eigen::Dynamic> const>
        E; ///< `|Element::Nodes| x |# elements|` element nodal indices
};

namespace detail {
template <CElement TElement>
class NodalKey
{
  public:
    using SelfType                 = NodalKey<TElement>;
    static int constexpr kVertices = TElement::AffineBaseType::kNodes;

    NodalKey(
        IndexVector<kVertices> const& cellVertices,
        IndexVector<kVertices> const& sortOrder,
        Eigen::Vector<math::Rational, kVertices> const& N)
        : mCellVertices(cellVertices), mSortOrder(sortOrder), mN(N), mSize()
    {
        // Remove vertices whose corresponding shape function is zero
        auto it = std::remove_if(mSortOrder.begin(), mSortOrder.end(), [this](Index o) {
            return mN[o] == 0;
        });
        // Count number of non-zero shape functions into Size
        mSize = static_cast<decltype(mSize)>(std::distance(mSortOrder.begin(), it));
    }

    bool operator==(SelfType const& rhs) const
    {
        // Sizes must match
        if (mSize != rhs.mSize)
            return false;
        for (auto i = 0u; i < mSize; ++i)
        {
            // Vertices involved in affine map must match
            Index const lhsVertex = mCellVertices[mSortOrder[i]];
            Index const rhsVertex = rhs.mCellVertices[rhs.mSortOrder[i]];
            if (lhsVertex != rhsVertex)
                return false;
            // Affine weights at matching vertices must match
            math::Rational const lhsN = mN[mSortOrder[i]];
            math::Rational const rhsN = rhs.mN[rhs.mSortOrder[i]];
            if (lhsN != rhsN)
                return false;
        }
        // Everything matches, (*this) and rhs must represent same node
        return true;
    }

    bool operator<(SelfType const& rhs) const
    {
        // Sort by size first
        if (mSize != rhs.mSize)
            return mSize < rhs.mSize;
        // Then sort by vertex indices
        for (auto i = 0; i < mSize; ++i)
        {
            Index const lhsVertex = mCellVertices[mSortOrder[i]];
            Index const rhsVertex = rhs.mCellVertices[rhs.mSortOrder[i]];
            if (lhsVertex != rhsVertex)
                return lhsVertex < rhsVertex;
        }
        // Then sort by coordinates
        for (auto i = 0; i < mSize; ++i)
        {
            math::Rational const lhsN = mN[mSortOrder[i]];
            math::Rational const rhsN = rhs.mN[rhs.mSortOrder[i]];
            if (lhsN != rhsN)
                return lhsN < rhsN;
        }
        // (*this) == rhs is true, so (*this) is not less than rhs
        return false;
    }

  private:
    IndexVector<kVertices> mCellVertices;        ///< Cell vertex indices
    IndexVector<kVertices> mSortOrder;           ///< Ordering of the cell vertices
    Eigen::Vector<math::Rational, kVertices> mN; ///< Node's affine shape function values
    int mSize; ///< Number of non-zero affine shape function values
};

} // namespace detail

template <CElement TElement, int Dims>
Mesh<TElement, Dims>::Mesh(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
{
    Construct(V, C);
}

template <CElement TElement, int Dims>
inline void Mesh<TElement, Dims>::Construct(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.Mesh.Construct");

    // Smart nodal indexing is only relevant for higher-order meshes
    if constexpr (kOrder == 1)
    {
        X = V;
        E = C;
        return;
    }
    else
    {
        using AffineElementType         = typename ElementType::AffineBaseType;
        auto constexpr kVerticesPerCell = AffineElementType::kNodes;

        static_assert(
            kDims >= ElementType::kDims,
            "Element TElement does not exist in Dims dimensions");
        assert(C.rows() == kVerticesPerCell);
        assert(V.rows() == kDims);

        using detail::NodalKey;
        using NodeMap = std::map<NodalKey<ElementType>, Index>;

        auto const numberOfCells    = C.cols();
        auto const numberOfVertices = V.cols();

        NodeMap nodeMap{};
        std::vector<Vector<kDims>> nodes{};
        nodes.reserve(static_cast<std::size_t>(numberOfVertices));

        // Construct mesh topology, i.e. assign mesh nodes to elements,
        // ensuring that adjacent elements share their common nodes.
        E.resize(ElementType::kNodes, numberOfCells);
        for (auto c = 0; c < numberOfCells; ++c)
        {
            IndexVector<kVerticesPerCell> const cellVertices = C.col(c);
            Matrix<kDims, kVerticesPerCell> const Xc = V(Eigen::placeholders::all, cellVertices);

            // Sort based on cell vertex index
            IndexVector<kVerticesPerCell> sortOrder{};
            std::iota(sortOrder.begin(), sortOrder.end(), 0);
            std::ranges::sort(sortOrder, [&](Index i, Index j) {
                return cellVertices[i] < cellVertices[j];
            });
            // Loop over nodes of element and create the node on first visit
            auto const nodalCoordinates = common::ToEigen(ElementType::Coordinates)
                                              .reshaped(ElementType::kDims, ElementType::kNodes)
                                              .template cast<math::Rational>() /
                                          ElementType::kOrder;
            for (auto i = 0; i < nodalCoordinates.cols(); ++i)
            {
                // Use exact rational arithmetic to evaluate affine element shape functions at the
                // node to get its exact affine coordinates
                auto const Xi = nodalCoordinates.col(i);
                auto const N  = AffineElementType::N(Xi);
                NodalKey<ElementType> const key{cellVertices, sortOrder, N};
                auto it                        = nodeMap.find(key);
                bool const bNodeAlreadyCreated = it != nodeMap.end();
                if (!bNodeAlreadyCreated)
                {
                    auto const nodeIdx     = static_cast<Index>(nodes.size());
                    Vector<kDims> const xi = Xc * N.template cast<Scalar>();
                    nodes.push_back(xi);
                    bool bInserted{};
                    std::tie(it, bInserted) = nodeMap.insert({key, nodeIdx});
                    assert(bInserted);
                }
                Index const node = it->second;
                E(i, c)          = node;
            }
        }
        // Collect node positions
        X = common::ToEigen(nodes);
    }
}

namespace detail {

template <CElement TElement, int Dims, int QuadratureOrder, class TDerivedX, class TDerivedE>
inline MatrixX
MeshQuadraturePoints(Eigen::MatrixBase<TDerivedX> const& X, Eigen::MatrixBase<TDerivedE> const& E)
{
    using ElementType           = TElement;
    using AffineElementType     = typename ElementType::AffineBaseType;
    using QuadratureRuleType    = typename ElementType::template QuadratureType<QuadratureOrder>;
    auto constexpr kQuadPts     = QuadratureRuleType::kPoints;
    auto const numberOfElements = E.cols();
    auto const XgRef            = common::ToEigen(QuadratureRuleType::points)
                           .reshaped(QuadratureRuleType::kDims + 1, kQuadPts);
    MatrixX Xg(Dims, numberOfElements * kQuadPts);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes                = E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = Dims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = X(Eigen::placeholders::all, vertices);
        for (auto g = 0; g < kQuadPts; ++g)
        {
            Xg.col(e * kQuadPts + g) = Ve * XgRef.col(g);
        }
    }
    return Xg;
}

template <CElement TElement, int Dims, int QuadratureOrder>
inline MatrixX MeshQuadratureWeights()
{
    using ElementType         = TElement;
    using QuadratureRuleType  = typename ElementType::template QuadratureType<QuadratureOrder>;
    auto constexpr kQuadPts   = QuadratureRuleType::kPoints;
    Vector<kQuadPts> const wg = common::ToEigen(QuadratureRuleType::weights);
    return wg;
}

} // namespace detail

template <CElement TElement, int Dims>
template <int QuadratureOrder>
inline MatrixX Mesh<TElement, Dims>::QuadraturePoints() const
{
    return detail::MeshQuadraturePoints<ElementType, kDims, QuadratureOrder>(X, E);
}

template <CElement TElement, int Dims>
template <int QuadratureOrder>
inline Vector<TElement::template QuadratureType<QuadratureOrder>::kPoints>
Mesh<TElement, Dims>::QuadratureWeights() const
{
    return detail::MeshQuadratureWeights<ElementType, kDims, QuadratureOrder>();
}

template <CElement TElement, int Dims>
inline LinearMeshView<TElement, Dims>::LinearMeshView(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
    : X(V), E(C)
{
    static_assert(TElement::kOrder == 1, "TElement must be a linear element");
    auto const nNodesPerElement = C.rows();
    if (nNodesPerElement != TElement::kNodes)
    {
        throw std::invalid_argument(
            fmt::format(
                "Expected {}x{} elements, but got {}x{}",
                TElement::kNodes,
                C.cols(),
                C.rows(),
                C.cols()));
    }
    if (V.rows() != Dims)
    {
        throw std::invalid_argument(
            fmt::format("Expected {}x{} nodes, but got {}x{}", Dims, V.cols(), V.rows(), V.cols()));
    }
    if (V.rows() < TElement::kDims)
    {
        throw std::invalid_argument(
            fmt::format(
                "Nodal coordinates must have dimensions > {} for the requested element",
                TElement::kDims));
    }
}

template <CElement TElement, int Dims>
template <int QuadratureOrder>
inline MatrixX LinearMeshView<TElement, Dims>::QuadraturePoints() const
{
    return detail::MeshQuadraturePoints<ElementType, kDims, QuadratureOrder>(X, E);
}

template <CElement TElement, int Dims>
template <int QuadratureOrder>
inline Vector<TElement::template QuadratureType<QuadratureOrder>::kPoints>
LinearMeshView<TElement, Dims>::QuadratureWeights() const
{
    return detail::MeshQuadratureWeights<ElementType, kDims, QuadratureOrder>();
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_MESH_H
