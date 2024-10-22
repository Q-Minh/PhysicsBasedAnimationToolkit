#ifndef PBAT_FEM_MESH_H
#define PBAT_FEM_MESH_H

#include "Concepts.h"
#include "Jacobian.h"

#include <algorithm>
#include <exception>
#include <map>
#include <numeric>
#include <pbat/Aliases.h>
#include <pbat/common/Eigen.h>
#include <pbat/math/Rational.h>
#include <pbat/profiling/Profiling.h>
#include <ranges>

namespace pbat {
namespace fem {

template <CElement TElement, int Dims>
struct Mesh
{
    using ElementType           = TElement;            ///< Underlying finite element type
    static int constexpr kDims  = Dims;                ///< Embedding dimensions of the mesh
    static int constexpr kOrder = ElementType::kOrder; ///< Shape function order

    Mesh() = default;
    /**
     * @brief Constructs a finite element mesh given some input geometric mesh. The cells of the
     * input mesh should list its vertices in Lagrange order.
     * @param V Dims x |#vertices| matrix of vertex positions
     * @param C Element::AffineBase::Vertices x |#cells| matrix of cell vertex indices into V
     */
    Mesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);

    /**
     * @brief Compute quadrature points in domain space for on this mesh.
     * @tparam QuadratureOrder
     * @return
     */
    template <int QuadratureOrder>
    MatrixX QuadraturePoints() const;
    /**
     * @brief Obtain quadrature weights on the reference element of this mesh
     * @tparam QuadratureOrder
     * @return
     */
    template <int QuadratureOrder>
    Vector<TElement::template QuadratureType<QuadratureOrder>::kPoints> QuadratureWeights() const;

    MatrixX X;      ///< Dims x |#nodes| nodal positions
    IndexMatrixX E; ///< Element::Nodes x |#elements| element nodal indices
};

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

template <CElement TElement, int Dims>
Mesh<TElement, Dims>::Mesh(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
{
    PBAT_PROFILE_NAMED_SCOPE("fem.Mesh.Construct");

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
            Matrix<kDims, kVerticesPerCell> const Xc         = V(Eigen::all, cellVertices);

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

template <CElement TElement, int Dims>
template <int QuadratureOrder>
inline MatrixX Mesh<TElement, Dims>::QuadraturePoints() const
{
    using AffineElementType     = typename ElementType::AffineBaseType;
    using QuadratureRuleType    = typename ElementType::template QuadratureType<QuadratureOrder>;
    auto constexpr kQuadPts     = QuadratureRuleType::kPoints;
    auto const numberOfElements = E.cols();
    auto const XgRef            = common::ToEigen(QuadratureRuleType::points)
                           .reshaped(QuadratureRuleType::kDims + 1, kQuadPts);
    MatrixX Xg(kDims, numberOfElements * kQuadPts);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes                = E.col(e);
        auto const vertices             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ           = kDims;
        auto constexpr kColsJ           = AffineElementType::kNodes;
        Matrix<kRowsJ, kColsJ> const Ve = X(Eigen::all, vertices);
        for (auto g = 0; g < kQuadPts; ++g)
        {
            Xg.col(e * kQuadPts + g) = Ve * XgRef.col(g);
        }
    }
    return Xg;
}

template <CElement TElement, int Dims>
template <int QuadratureOrder>
inline Vector<TElement::template QuadratureType<QuadratureOrder>::kPoints>
Mesh<TElement, Dims>::QuadratureWeights() const
{
    using QuadratureRuleType  = typename ElementType::template QuadratureType<QuadratureOrder>;
    auto constexpr kQuadPts   = QuadratureRuleType::kPoints;
    Vector<kQuadPts> const wg = common::ToEigen(QuadratureRuleType::weights);
    return wg;
}

} // namespace fem
} // namespace pbat

#endif // PBAT_FEM_MESH_H