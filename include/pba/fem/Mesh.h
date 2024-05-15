#ifndef PBA_CORE_FEM_MESH_H
#define PBA_CORE_FEM_MESH_H

#include "Concepts.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/common/Hash.h"
#include "pba/math/Rational.h"

#include <Eigen/QR>
#include <algorithm>
#include <map>
#include <numeric>
#include <ranges>

namespace pba {
namespace fem {

template <CElement TElement, int Dims>
struct Mesh
{
    using ElementType          = TElement; ///< Underlying finite element type
    static int constexpr kDims = Dims;     ///< Embedding dimensions of the mesh

    Mesh() = default;
    /**
     * @brief Constructs a finite element mesh given some input geometric mesh. The cells of the
     * input mesh should list its vertices in Lagrange order.
     * @param V Dims x |#vertices| matrix of vertex positions
     * @param C Element::AffineBase::Vertices x |#cells| matrix of cell vertex indices into V
     */
    Mesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);

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
                                          .cast<math::Rational>() /
                                      ElementType::kOrder;
        for (auto i = 0; i < nodalCoordinates.cols(); ++i)
        {
            // Use exact rational arithmetic to evaluate affine element shape functions at the node
            // to get its exact affine coordinates
            auto const Xi = nodalCoordinates.col(i);
            auto const N  = AffineElementType::N(Xi);
            NodalKey<ElementType> const key{cellVertices, sortOrder, N};
            auto it                        = nodeMap.find(key);
            bool const bNodeAlreadyCreated = it != nodeMap.end();
            if (!bNodeAlreadyCreated)
            {
                auto const nodeIdx     = static_cast<Index>(nodes.size());
                Vector<kDims> const xi = Xc * N.cast<Scalar>();
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

    // TODO: Move this code to any LinearOperator acting on an fem::Mesh that needs to evaluate
    // jacobians at quadrature points. auto const NumberOfElements = E.cols();
    // // Compute element mappings
    // J.resize(TElement::Dims, Dims * NumberOfElements);
    // Jinv.resize(Dims, TElement::Dims * NumberOfElements);
    // detJ.resize(NumberOfElements);
    // for (auto e = 0; e < NumberOfElements; ++e)
    // {
    //     Matrix<Dims, TElement::Nodes> const CellNodalPositions = X(Eigen::all, E.col(e));
    //     // Compute jacobian of map from reference element to domain
    //     Matrix<Dims, TElement::Dims> const Jc = TElement::Jacobian()
    //     J.block(0, c * TElement::Dims, Dims, TElement::Dims) = Jc;
    // }
    // for (auto c = 0; c < NumberOfCells; ++c)
    // {
    //     // Jinv = (Jc^T Jc)^{-1} Jc^T (i.e. normal equations) using stable QR factorization
    //     auto const Jc = J.block(0, c * TElement::Dims, Dims, TElement::Dims);
    //     Matrix<TElement::Dims, Dims> const JcInv =
    //         (Jc.transpose() * Jc).fullPivHouseholderQr().solve(Jc.transpose());
    //     Jinv.block(0, c * Dims, TElement::Dims, Dims) = JcInv;
    // }
    // for (auto c = 0; c < NumberOfCells; ++c)
    // {
    //     // In the general case, we want to compute sqrt(det(J^T J)) to measure the
    //     // rate of generalized volume change induced by the local map J. To do this,
    //     // let J = U*E*V^T, then J^T J = V*E*E*V^T, such that det(J^T J) = det(E)^2.
    //     // Because E is diagonal, det(E) = \prod_i \sigma_i where \sigma_i is the
    //     // i^{th} singular value J. We thus have that sqrt(det(J^T J)) = \prod_i \sigma_i.
    //     // NOTE: Negative determinant means inverted element.
    //     auto const Jc = J.block(0, c * TElement::Dims, Dims, TElement::Dims);
    //     detJ(c)       = Jc.jacobiSvd().singularValues().prod();
    // }
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_MESH_H