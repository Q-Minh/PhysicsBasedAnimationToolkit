#ifndef PBAT_GRAPH_DISJOINTSETUNIONFIND_H
#define PBAT_GRAPH_DISJOINTSETUNIONFIND_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"

#include <cassert>

namespace pbat::graph {

/**
 * @brief Disjoint Set Union-Find data structure
 * @tparam TIndex Index type used in the disjoint set union-find structure.
 *
 * This class implements a disjoint set union-find data structure with path compression and union by
 * rank. It is used to efficiently manage and merge disjoint sets, which is useful in various graph
 * algorithms.
 */
template <common::CIndex TIndex = Index>
class DisjointSetUnionFind
{
  public:
    using IndexType = TIndex; ///< Index type used in the disjoint set union-find structure.

    DisjointSetUnionFind() = default;
    /**
     * @brief Construct a new Disjoint Set Union-Find object
     * @param n Number of vertices
     */
    DisjointSetUnionFind(IndexType n);
    /**
     * @brief Reserve memory for `n` vertices and reset the structure
     * @param n Number of vertices
     */
    void Prepare(IndexType n);
    /**
     * @brief Find the root of the set containing vertex `u`
     * @param u Vertex index
     * @return Root of the set containing vertex `u`
     */
    IndexType Find(IndexType u);
    /**
     * @brief Merge the sets containing vertices `u` and `v`
     * @param u Vertex index of the first set
     * @param v Vertex index of the second set
     * @return Root of the merged tree
     */
    IndexType Union(IndexType u, IndexType v);
    /**
     * @brief Find the root of the set containing vertex `u` without path compression
     * @param u Vertex index
     * @return Root of the set containing vertex `u`
     */
    IndexType Root(IndexType u) const;
    /**
     * @brief Find the size of the set containing vertex `u`
     * @param u Vertex index
     * @return Size of the set containing vertex `u`
     */
    IndexType Size(IndexType u) const;
    /**
     * @brief Get the number of vertices in the disjoint set union-find structure
     * @return Number of vertices
     */
    IndexType NumVertices() const;

  private:
    Eigen::Vector<IndexType, Eigen::Dynamic> mParent;
    Eigen::Vector<IndexType, Eigen::Dynamic> mRank;
    Eigen::Vector<IndexType, Eigen::Dynamic> mSize;
};

template <common::CIndex TIndex>
DisjointSetUnionFind<TIndex>::DisjointSetUnionFind(IndexType n) : DisjointSetUnionFind<TIndex>()
{
    Prepare(n);
}

template <common::CIndex TIndex>
void DisjointSetUnionFind<TIndex>::Prepare(IndexType n)
{
    mParent = Eigen::Vector<IndexType, Eigen::Dynamic>::LinSpaced(n, IndexType(0), n - 1);
    mRank.setZero(n);
    mSize.setOnes(n);
}

template <common::CIndex TIndex>
TIndex DisjointSetUnionFind<TIndex>::Find(IndexType u)
{
    if (mParent[u] != u)
        mParent[u] = Find(mParent[u]); // Path compression
    return mParent[u];
}

template <common::CIndex TIndex>
TIndex DisjointSetUnionFind<TIndex>::Union(IndexType u, IndexType v)
{
    assert(u != v && "Cannot union the same vertex with itself.");
    // Find roots of each node
    IndexType const ru = Find(u);
    IndexType const rv = Find(v);
    // Make tree with smaller height a subtree of the other tree.
    // This effectively minimizes the depth of the resulting merged tree.
    if (mRank[ru] > mRank[rv])
    {
        mParent[rv] = ru;
        mSize[ru] += mSize[rv];
        mSize[rv] = IndexType(0);
    }
    else
    {
        mParent[ru] = rv;
        mSize[rv] += mSize[ru];
        mSize[ru] = IndexType(0);
    }
    // If both trees had the same depth, then
    // making one tree a subtree of the other increased
    // the depth of the merged tree by 1.
    if (mRank[ru] == mRank[rv])
    {
        ++mRank[rv];
    }
    return (mParent[ru] == ru) ? ru : rv;
}

template <common::CIndex TIndex>
inline TIndex DisjointSetUnionFind<TIndex>::Root(IndexType u) const
{
    while (mParent[u] != u)
        u = mParent[u];
    return u;
}

template <common::CIndex TIndex>
TIndex DisjointSetUnionFind<TIndex>::Size(IndexType u) const
{
    return mSize[Root(u)];
}

template <common::CIndex TIndex>
TIndex DisjointSetUnionFind<TIndex>::NumVertices() const
{
    return static_cast<TIndex>(mSize.size());
}

} // namespace pbat::graph

#endif // PBAT_GRAPH_DISJOINTSETUNIONFIND_H