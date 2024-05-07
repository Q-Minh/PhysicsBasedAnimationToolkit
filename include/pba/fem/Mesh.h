#ifndef PBA_CORE_FEM_MESH_H
#define PBA_CORE_FEM_MESH_H

#include "Concepts.h"
#include "pba/aliases.h"
#include "pba/common/Eigen.h"
#include "pba/common/Hash.h"

#include <Eigen/QR>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <unordered_map>

namespace pba {
namespace fem {

// template <int Order>
// struct Line;

// template <int Order>
// struct Triangle;

// template <int Order>
// struct Quadrilateral;

// template <int Order>
// struct Tetrahedron;

// template <int Order>
// struct Hexahedron;

// #include <array>

// template <>
// struct Tetrahedron<1>
// {
//     static int constexpr Order = 1;
//     static int constexpr Dims  = 3;
//     static int constexpr Nodes = 4;
//     static int constexpr Vertices = 4;
//     static std::array<int, Nodes * Dims> constexpr Coordinates =
//         {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};

//     Vector<Nodes> N(Vector<Dims> const& X) const;
//     Matrix<Nodes, Dims> GradN(Vector<Dims> const& X) const;
// };

template <Element TElement, int Dims>
struct Mesh
{
    using ElementType         = TElement;
    static int constexpr Dims = Dims;

    Mesh() = default;
    Mesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& C);

    MatrixX X;      ///< Dims x |Nodes| nodal positions
    IndexMatrixX E; ///< |#nodes per element| x |Elements| element nodal indices
    MatrixX J;    ///< Element::Dims x Dims element linear mappings stacked horizontally |Elements|
                  ///< times, i.e. the reference -> domain transforms
    MatrixX Jinv; ///< Dims x Element::Dims inverse element linear mappings stacked horizontally
                  ///< |Elements| times, i.e. the domain -> reference transforms
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

template <Element TElement, int Dims>
Mesh<TElement, Dims>::Mesh(
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& C)
{
    assert(C.rows() == TElement::Vertices);

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

    // Compute element mappings
    J.resize(TElement::Dims, Dims * NumberOfCells);
    Jinv.resize(Dims, TElement::Dims * NumberOfCells);
    for (auto c = 0; c < NumberOfCells; ++c)
    {
        Matrix<Dims, TElement::Vertices> const CellVertexPositions = V(Eigen::all, C.col(c));
        // Compute affine map from reference element to domain
        Matrix<TElement::Dims, Dims> const Jc      = TElement::AffineMap(CellVertexPositions);
        J.block(0, c * Dims, TElement::Dims, Dims) = Jc;
        // Jinv = (Jc^T Jc)^{-1} Jc^T (i.e. normal equations) using stable QR factorization
        Matrix<Dims, TElement::Dims> const JcInv =
            (Jc.transpose() * Jc).fullPivHouseholderQr().solve(Jc.transpose());
        Jinv.block(0, c * TElement::Dims, Dims, TElement::Dims) = JcInv;
    }
}

} // namespace fem
} // namespace pba

#endif // PBA_CORE_FEM_MESH_H