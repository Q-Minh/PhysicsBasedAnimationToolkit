#ifndef PBA_CORE_FEM_MESH_H
#define PBA_CORE_FEM_MESH_H

#include "Concepts.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/common/Hash.h"
#include "pba/math/Rational.h"

#include <Eigen/QR>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <unordered_map>

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
struct NodalKey
{
    NodalKey(
        IndexVector<TElement::Vertices> const& InCellVertices,
        IndexVector<TElement::Vertices> const& InSortOrder,
        IndexVector<TElement::Vertices> const& InCoordinates)
        : CellVertices(InCellVertices), SortOrder(InSortOrder), Coordinates(InCoordinates), Size()
    {
        // Remove vertices whose corresponding coordinate is zero
        auto it = std::remove_if(SortOrder.begin(), SortOrder.end(), [](Index o) {
            return Coordinates[o] == 0;
        });
        // Count number of non-zero coordinates into Size
        Size = std::distance(SortOrder.begin(), it);
    }

    bool operator==(NodalKey const& rhs) const
    {
        if (Size != rhs.Size)
            return false;

        for (auto i = 0u; i < Size; ++i)
        {
            Index const LhsVertex = CellVertices[SortOrder[i]];
            Index const RhsVertex = rhs.CellVertices[rhs.SortOrder[i]];
            if (LhsVertex != RhsVertex)
                return false;
            Index const LhsCoordinate = Coordinates[SortOrder[i]];
            Index const RhsCoordinate = rhs.Coordinates[rhs.SortOrder[i]];
            if (LhsCoordinate != RhsCoordinate)
                return false;
        }
        return true;
    }

    IndexVector<TElement::Vertices> CellVertices; ///< Cell vertex indices
    IndexVector<TElement::Vertices> SortOrder;    ///< Ordering of the cell vertices
    IndexVector<TElement::Vertices> Coordinates;  ///< Node's integer coordinates
    int Size;
};

template <Element TElement>
struct NodalKeyHash
{
    std::size_t operator()(NodalKey<TElement> const& Key) const
    {
        // Hopefully, this is an acceptable hash function for a NodalKey
        std::size_t seed{Key.Size};
        for (auto i = 0u; i < Key.Size; ++i)
        {
            Index const Vertex = Key.Cell[Key.SortOrder[i]];
            common::hash_combine_accumulate(seed, Vertex);
            Index const Coordinate = Key.Coordinates[Key.SortOrder[i]];
            common::hash_combine_accumulate(seed, Coordinate);
        }
        return seed;
    }
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

    auto const NumberOfCells    = C.cols();
    auto const NumberOfVertices = V.cols();

    using NodeMap = std::unordered_map<NodalKey<TElement>, Index, NodalKeyHash<TElement>>;
    NodeMap N{};
    N.reserve(NumberOfCells * TElement::Nodes);
    std::vector<Vector<Dims>> Nodes{};
    Nodes.reserve(NumberOfVertices);

    // Construct mesh topology, i.e. assign mesh nodes to elements,
    // ensuring that adjacent elements share their common nodes.
    E.resize(TElement::Nodes, NumberOfCells);
    for (auto c = 0; c < NumberOfCells; ++c)
    {
        IndexVector<TElement::Vertices> const CellVertices         = C.col(c);
        Matrix<Dims, TElement::Vertices> const CellVertexPositions = V(Eigen::all, CellVertices);

        // Sort based on cell vertex index
        IndexVector<TElement::Vertices> SortOrder{};
        std::iota(SortOrder.begin(), SortOrder.end(), 0);
        std::ranges::sort(SortOrder, [&](Index i, Index j) {
            return CellVertices[i] < CellVertices[j];
        });
        // Loop over nodes of element and create the node on first visit
        auto const NodalCoordinates =
            common::ToEigen(TElement::Coordinates).reshape(TElement::Dims, TElement::Nodes);
        for (auto i = 0; i < TElement::Nodes; ++i)
        {
            // Use exact rational arithmetic to evaluate affine element shape functions at the node
            // to get its exact coordinates
            Eigen::Vector<math::Rational, TElement::Dims> Xi{};
            for (auto j = 0; j < NodalCoordinates.rows(); ++j)
                Xi(j) = math::Rational(NodalCoordinates(j,i), TElement::Order);
            Eigen::Vector<math::Rational, AffineElement::Nodes> N = AffineElement::N(Xi);
            for (auto j = 0; j < N.size(); ++j)
                N(j).rebase(TElement::Order);
            IndexVector<TElement::Vertices> Coordinates{};
            Coordinates(0) = TElement::Order - NodalCoordinates.col(i).sum();
            Coordinates.segment(1, TElement::Dims) = NodalCoordinates.col(i);
            NodalKey<TElement> const Key{CellVertices, SortOrder, Coordinates};
            auto it                       = N.find(Key);
            bool const NodeAlreadyCreated = it != N.end();
            if (!NodeAlreadyCreated)
            {
                auto const NodeIdx = static_cast<Index>(Nodes.size());
                auto const RealCoordinates =
                    Coordinates.cast<Scalar>() / static_cast<Scalar>(TElement::Order);
                // WARNING: Only works for simplex elements.
                // TODO: Add customization point for non-simplex elements (i.e. quadrilateral,
                // hexahedron)
                assert(CellVertexPositions.cols() == RealCoordinates.rows());
                Vector<Dims> const Xi = CellVertexPositions * RealCoordinates;
                Nodes.push_back(Xi);
                bool inserted{};
                std::tie(it, inserted) = N.insert({Key, NodeIdx});
                assert(inserted);
            }
            Index const Node = it->second;
            E(i, c)          = Node;
        }
    }
    // Collect node positions
    X = common::ToEigen(Nodes);

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