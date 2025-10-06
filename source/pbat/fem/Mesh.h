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
#include "pbat/io/Archive.h"
#include "pbat/math/Rational.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <map>
#include <numeric>
#include <ranges>

namespace pbat::fem {

/**
 * @brief A generic stateful finite element mesh representation.
 *
 * @tparam TElement Type satisfying concept CElement
 * @tparam Dims Embedding dimensions of the mesh
 * @tparam TScalar Floating point type, defaults to Scalar
 * @tparam TIndex Index type, defaults to Index
 */
template <
    CElement TElement,
    int Dims,
    common::CFloatingPoint TScalar = Scalar,
    common::CIndex TIndex          = Index>
struct Mesh
{
    using ElementType                     = TElement; ///< Underlying finite element type
    using ScalarType                      = TScalar;  ///< Floating point type
    using IndexType                       = TIndex;   ///< Index type
    static int constexpr kDims            = Dims;     ///< Embedding dimensions of the mesh
    static int constexpr kOrder           = ElementType::kOrder; ///< Shape function order
    static int constexpr kNodesPerElement = ElementType::kNodes; ///< Number of nodes per element
    using NodeMatrix =
        Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic>; ///< Node positions matrix type
    using ElementMatrix =
        Eigen::Matrix<IndexType, kNodesPerElement, Eigen::Dynamic>; ///< Element indices matrix type

    Mesh() = default;
    /**
     * @brief Constructs a finite element mesh given some input geometric mesh.
     *
     * @warning The cells of the input mesh should list its vertices in Lagrange order.
     * @tparam TDerivedV Type of the vertex positions matrix
     * @tparam TDerivedC Type of the cell vertex indices matrix
     * @param V `Dims x |# vertices|` matrix of vertex positions
     * @param C `|Element::AffineBase::Vertices| x |# cells|` matrix of cell vertex indices into V
     */
    template <class TDerivedV, class TDerivedC>
    Mesh(Eigen::MatrixBase<TDerivedV> const& V, Eigen::DenseBase<TDerivedC> const& C);
    /**
     * @brief Constructs a finite element mesh given some input geometric mesh.
     *
     * @warning The cells of the input mesh should list its vertices in Lagrange order.
     * @tparam TDerivedV Type of the vertex positions matrix
     * @tparam TDerivedC Type of the cell vertex indices matrix
     * @param V `Dims x |# vertices|` matrix of vertex positions
     * @param C `|Element::AffineBase::Vertices| x |# cells|` matrix of cell vertex indices into V
     */
    template <class TDerivedV, class TDerivedC>
    void Construct(Eigen::MatrixBase<TDerivedV> const& V, Eigen::DenseBase<TDerivedC> const& C);
    /**
     * @brief Compute quadrature points in domain space on this mesh.
     * @tparam QuadratureOrder Quadrature order
     * @return `kDims x |# element quad.pts.|` matrix of quadrature points
     */
    template <int QuadratureOrder>
    auto QuadraturePoints() const -> Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
    /**
     * @brief Obtain quadrature weights on the reference element of this mesh
     * @tparam QuadratureOrder Quadrature order
     * @param kPoints Number of quadrature points, defaults to `Element::template
     * QuadratureType<QuadratureOrder>::kPoints`. This template parameter should not be specified,
     * it is there to enhance readability.
     * @return `|# element quad.pts.|` vector of quadrature weights
     */
    template <
        int QuadratureOrder,
        int kPoints = TElement::template QuadratureType<QuadratureOrder, ScalarType>::kPoints>
    auto QuadratureWeights() const -> Eigen::Vector<ScalarType, kPoints>;

    /**
     * @brief Serialize to HDF5 group
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize from HDF5 group
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive const& archive);

    NodeMatrix X;    ///< `kDims x |# nodes|` nodal positions
    ElementMatrix E; ///< `|Element::Nodes| x |# elements|` element nodal indices
};

namespace detail {

template <CElement TElement, common::CIndex TIndex>
class NodalKey
{
  public:
    using SelfType                 = NodalKey<TElement, TIndex>; ///< Self type for convenience
    static int constexpr kVertices = TElement::AffineBaseType::kNodes;

    NodalKey(
        Eigen::Vector<TIndex, kVertices> const& cellVertices,
        Eigen::Vector<TIndex, kVertices> const& sortOrder,
        Eigen::Vector<math::Rational, kVertices> const& N)
        : mCellVertices(cellVertices), mSortOrder(sortOrder), mN(N), mSize()
    {
        // Remove vertices whose corresponding shape function is zero
        auto it = std::remove_if(mSortOrder.begin(), mSortOrder.end(), [this](TIndex o) {
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
            TIndex const lhsVertex = mCellVertices[mSortOrder[i]];
            TIndex const rhsVertex = rhs.mCellVertices[rhs.mSortOrder[i]];
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
            TIndex const lhsVertex = mCellVertices[mSortOrder[i]];
            TIndex const rhsVertex = rhs.mCellVertices[rhs.mSortOrder[i]];
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
    Eigen::Vector<TIndex, kVertices> mCellVertices; ///< Cell vertex indices
    Eigen::Vector<TIndex, kVertices> mSortOrder;    ///< Ordering of the cell vertices
    Eigen::Vector<math::Rational, kVertices> mN;    ///< Node's affine shape function values
    int mSize; ///< Number of non-zero affine shape function values
};

} // namespace detail

template <CElement TElement, int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedV, class TDerivedC>
inline Mesh<TElement, Dims, TScalar, TIndex>::Mesh(
    Eigen::MatrixBase<TDerivedV> const& V,
    Eigen::DenseBase<TDerivedC> const& C)
{
    Construct(V, C);
}

template <CElement TElement, int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedV, class TDerivedC>
inline void Mesh<TElement, Dims, TScalar, TIndex>::Construct(
    Eigen::MatrixBase<TDerivedV> const& V,
    Eigen::DenseBase<TDerivedC> const& C)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.fem.Mesh.Construct");
    static_assert(
        kDims >= ElementType::kDims,
        "Element TElement does not exist in Dims dimensions");
    static_assert(
        std::is_same_v<TScalar, typename TDerivedV::Scalar>,
        "Vertex positions matrix V must have the same scalar type as TScalar");
    static_assert(
        std::is_same_v<TIndex, typename TDerivedC::Scalar>,
        "Cell vertex indices matrix C must have the same index type as TIndex");

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
        assert(C.rows() == kVerticesPerCell);
        assert(V.rows() == kDims);

        using NodalKey = detail::NodalKey<TElement, TIndex>;
        using NodeMap  = std::map<NodalKey, TIndex>;
        using DVector  = Eigen::Vector<TScalar, kDims>;

        auto const numberOfCells    = C.cols();
        auto const numberOfVertices = V.cols();

        NodeMap nodeMap{};
        std::vector<DVector> nodes{};
        nodes.reserve(static_cast<std::size_t>(numberOfVertices));

        // Construct mesh topology, i.e. assign mesh nodes to elements,
        // ensuring that adjacent elements share their common nodes.
        E.resize(ElementType::kNodes, numberOfCells);
        for (auto c = 0; c < numberOfCells; ++c)
        {
            Eigen::Vector<TIndex, kVerticesPerCell> cellVertices = C.col(c);
            Eigen::Matrix<TScalar, kDims, kVerticesPerCell> const Xc =
                V(Eigen::placeholders::all, cellVertices);

            // Sort based on cell vertex index
            decltype(cellVertices) sortOrder{};
            std::iota(sortOrder.begin(), sortOrder.end(), TIndex(0));
            std::ranges::sort(sortOrder, [&](TIndex i, TIndex j) {
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
                NodalKey const key{cellVertices, sortOrder, N};
                auto it                        = nodeMap.find(key);
                bool const bNodeAlreadyCreated = it != nodeMap.end();
                if (!bNodeAlreadyCreated)
                {
                    auto const nodeIdx                     = static_cast<TIndex>(nodes.size());
                    Eigen::Vector<TScalar, kDims> const xi = Xc * N.template cast<TScalar>();
                    nodes.push_back(xi);
                    bool bInserted{};
                    std::tie(it, bInserted) = nodeMap.insert({key, nodeIdx});
                    assert(bInserted);
                }
                TIndex const node = it->second;
                E(i, c)           = node;
            }
        }
        // Collect node positions
        X = common::ToEigen(nodes);
    }
}

template <CElement TElement, int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
template <int QuadratureOrder>
inline Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>
Mesh<TElement, Dims, TScalar, TIndex>::QuadraturePoints() const
{
    using QuadraturePointPositions = Eigen::Matrix<TScalar, Eigen::Dynamic, Eigen::Dynamic>;
    using AffineElementType        = typename ElementType::AffineBaseType;
    using QuadratureRuleType =
        typename ElementType::template QuadratureType<QuadratureOrder, TScalar>;
    auto constexpr kQuadPts     = QuadratureRuleType::kPoints;
    auto const numberOfElements = E.cols();
    auto const XgRef            = common::ToEigen(QuadratureRuleType::points)
                           .reshaped(QuadratureRuleType::kDims + 1, kQuadPts);
    QuadraturePointPositions Xg(Dims, numberOfElements * kQuadPts);
    for (auto e = 0; e < numberOfElements; ++e)
    {
        auto const nodes                                = E.col(e);
        auto const vertices                             = nodes(ElementType::Vertices);
        auto constexpr kRowsJ                           = Dims;
        auto constexpr kColsJ                           = AffineElementType::kNodes;
        Eigen::Matrix<TScalar, kRowsJ, kColsJ> const Ve = X(Eigen::placeholders::all, vertices);
        for (auto g = 0; g < kQuadPts; ++g)
        {
            Xg.col(e * kQuadPts + g) = Ve * XgRef.col(g);
        }
    }
    return Xg;
}

template <CElement TElement, int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
template <int QuadratureOrder, int kPoints>
inline Eigen::Vector<TScalar, kPoints>
Mesh<TElement, Dims, TScalar, TIndex>::QuadratureWeights() const
{
    using QuadratureRuleType =
        typename ElementType::template QuadratureType<QuadratureOrder, TScalar>;
    auto constexpr kQuadPts = QuadratureRuleType::kPoints;
    static_assert(
        kQuadPts == kPoints,
        "kPoints must match the number of quadrature points for the given QuadratureOrder");
    Eigen::Vector<TScalar, kQuadPts> const wg = common::ToEigen(QuadratureRuleType::weights);
    return wg;
}

template <CElement TElement, int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
void Mesh<TElement, Dims, TScalar, TIndex>::Serialize(io::Archive& archive) const
{
    io::Archive meshArchive = archive["pbat.fem.Mesh"];
    meshArchive.WriteData("X", X);
    meshArchive.WriteData("E", E);
    meshArchive.WriteMetaData("kDims", kDims);
    meshArchive.WriteMetaData("kOrder", kOrder);
}

template <CElement TElement, int Dims, common::CFloatingPoint TScalar, common::CIndex TIndex>
void Mesh<TElement, Dims, TScalar, TIndex>::Deserialize(io::Archive const& archive)
{
    io::Archive const meshArchive = archive["pbat.fem.Mesh"];
    int const kDimsRead           = meshArchive.ReadMetaData<int>("kDims");
    int const kOrderRead          = meshArchive.ReadMetaData<int>("kOrder");
    if (kDimsRead != kDims)
    {
        throw std::runtime_error(
            "pbat::fem::Mesh::Deserialize(): kDims in archive does not match template parameter "
            "kDims");
    }
    if (kOrderRead != kOrder)
    {
        throw std::runtime_error(
            "pbat::fem::Mesh::Deserialize(): kOrder in archive does not match template parameter "
            "kOrder");
    }
    X = meshArchive.ReadData<NodeMatrix>("X");
    E = meshArchive.ReadData<ElementMatrix>("E");
    if (E.rows() != ElementType::kNodes)
    {
        throw std::runtime_error(
            "pbat::fem::Mesh::Deserialize(): Number of rows in E does not match Element::kNodes");
    }
}

} // namespace pbat::fem

#endif // PBAT_FEM_MESH_H
