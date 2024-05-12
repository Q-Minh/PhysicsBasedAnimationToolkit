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

template <Element TElement, int Dims>
struct Mesh
{
    using ElementType         = TElement;
    static int constexpr Dims = Dims;

    Mesh() = default;
    Mesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);

    MatrixX X;      ///< Dims x |Nodes| nodal positions
    IndexMatrixX E; ///< |#nodes per element| x |Elements| element nodal indices
};

template <Element TElement>
class NodalKey
{
  public:
    NodalKey(
        IndexVector<TElement::Vertices> const& cellVertices,
        IndexVector<TElement::Vertices> const& sortOrder,
        Eigen::Vector<math::Rational, TElement::Vertices> const& N)
        : mCellVertices(cellVertices), mSortOrder(sortOrder), mN(N), mSize()
    {
        // Remove vertices whose corresponding shape function is zero
        auto it = std::remove_if(mSortOrder.begin(), mSortOrder.end(), [this](Index o) {
            return mN[o] == 0;
        });
        // Count number of non-zero shape functions into Size
        mSize = std::distance(mSortOrder.begin(), it);
    }

    bool operator==(NodalKey const& rhs) const
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
            Index const lhsN = mN[mSortOrder[i]];
            Index const rhsN = rhs.mN[rhs.mSortOrder[i]];
            if (lhsN != rhsN)
                return false;
        }
        // Everything matches, (*this) and rhs must represent same node
        return true;
    }

    bool operator<(NodalKey const& rhs) const
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
            Index const lhsN = mN[mSortOrder[i]];
            Index const rhsN = rhs.mN[rhs.mSortOrder[i]];
            if (lhsN != rhsN)
                return lhsN < rhsN;
        }
        // (*this) == rhs is true, so (*this) is not less than rhs
        return false;
    }

  private:
    IndexVector<TElement::Vertices> mCellVertices;        ///< Cell vertex indices
    IndexVector<TElement::Vertices> mSortOrder;           ///< Ordering of the cell vertices
    Eigen::Vector<math::Rational, TElement::Vertices> mN; ///< Node's affine shape function values
    int mSize; ///< Number of non-zero affine shape function values
};

// WARNING: Do not use, class is not yet usable.
template <Element TElement, int Dims>
Mesh<TElement, Dims>::Mesh(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
{
    static_assert(Dims >= TElement::Dims, "Element TElement does not exist in Dims dimensions");
    assert(C.rows() == TElement::Vertices);

    using AffineElement = TElement::AffineBase;
    using NodeMap       = std::map<NodalKey<TElement>, Index>;

    auto const numberOfCells    = C.cols();
    auto const numberOfVertices = V.cols();

    NodeMap nodeMap{};
    std::vector<Vector<Dims>> nodes{};
    nodes.reserve(numberOfVertices);

    // Construct mesh topology, i.e. assign mesh nodes to elements,
    // ensuring that adjacent elements share their common nodes.
    E.resize(TElement::Nodes, numberOfCells);
    for (auto c = 0; c < numberOfCells; ++c)
    {
        IndexVector<TElement::Vertices> const cellVertices = C.col(c);
        Matrix<Dims, TElement::Vertices> const Xe          = V(Eigen::all, cellVertices);

        // Sort based on cell vertex index
        IndexVector<TElement::Vertices> sortOrder{};
        std::iota(sortOrder.begin(), sortOrder.end(), 0);
        std::ranges::sort(sortOrder, [&](Index i, Index j) {
            return cellVertices[i] < cellVertices[j];
        });
        // Loop over nodes of element and create the node on first visit
        auto const nodalCoordinates =
            common::ToEigen(TElement::Coordinates).reshape(TElement::Dims, TElement::Nodes);
        for (auto i = 0; i < nodalCoordinates.cols(); ++i)
        {
            // Use exact rational arithmetic to evaluate affine element shape functions at the node
            // to get its exact affine coordinates
            Eigen::Vector<math::Rational, TElement::Dims> const Xi =
                nodalCoordinates.col(i).cast<math::Rational>() / TElement::Order;
            auto const N = AffineElement::N(Xi);
            NodalKey<TElement> const key{cellVertices, sortOrder, N};
            auto it                        = nodeMap.find(key);
            bool const bNodeAlreadyCreated = it != nodeMap.end();
            if (!bNodeAlreadyCreated)
            {
                auto const nodeIdx    = static_cast<Index>(nodes.size());
                Vector<Dims> const Xi = Xe * N.cast<Scalar>();
                nodes.push_back(Xi);
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