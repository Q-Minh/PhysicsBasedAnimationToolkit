#ifndef PBAT_GRAPH_BREADTHFIRSTSEARCH_H
#define PBAT_GRAPH_BREADTHFIRSTSEARCH_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/profiling/Profiling.h"

#include <queue>

namespace pbat::graph {

template <common::CIndex TIndex = Index>
struct BreadthFirstSearch
{
    using IndexType = TIndex;                ///< Index type used in the breadth-first search
    using QueueType = std::queue<IndexType>; ///< Queue type used for BFS

    BreadthFirstSearch() = default;
    /**
     * @brief Construct a new Breadth First Search object
     * @param n Number of vertices in the graph
     */
    BreadthFirstSearch(Eigen::Index n);
    /**
     * @brief Reserve memory for `n` vertices
     * @param n Number of vertices in the graph
     */
    void Reserve(Eigen::Index n);
    /**
     * @brief Perform depth-first search on the graph
     * @tparam FVisit Callable type with signature `void(IndexType)`
     * @tparam TDerivedP Type of the pointer vector (adjacency list start indices
     * for each vertex)
     * @tparam TDerivedAdj Type of the adjacency list (vector of vertex indices)
     * @param ptr Pointer to the start of each vertex's adjacency list
     * @param adj Adjacency list of the graph
     * @param start Starting vertex index
     * @param fVisit Function to call for each visited vertex
     */
    template <class FVisit, class TDerivedP, class TDerivedAdj>
    void operator()(
        Eigen::DenseBase<TDerivedP> const& ptr,
        Eigen::DenseBase<TDerivedAdj> const& adj,
        TIndex start,
        FVisit fVisit);

    Eigen::Vector<bool, Eigen::Dynamic> visited; ///< `|# vertices| x 1` visited mask
    QueueType queue;                             ///< BFS search queue
};

template <common::CIndex TIndex>
inline BreadthFirstSearch<TIndex>::BreadthFirstSearch(Eigen::Index n)
{
    Reserve(n);
}

template <common::CIndex TIndex>
inline void BreadthFirstSearch<TIndex>::Reserve(Eigen::Index n)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.BreadthFirstSearch.Reserve");
    visited.resize(n);
}

template <common::CIndex TIndex>
template <class FVisit, class TDerivedP, class TDerivedAdj>
inline void BreadthFirstSearch<TIndex>::operator()(
    Eigen::DenseBase<TDerivedP> const& ptr,
    Eigen::DenseBase<TDerivedAdj> const& adj,
    TIndex start,
    FVisit fVisit)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.BreadthFirstSearch");
    visited.setConstant(false);
    queue.push(start);
    while (not queue.empty())
    {
        IndexType u = queue.front();
        queue.pop();
        if (not visited[u])
        {
            fVisit(u);
            visited[u]  = true;
            auto kBegin = ptr[u];
            auto kEnd   = ptr[u + 1];
            for (auto k = kBegin; k < kEnd; ++k)
            {
                IndexType v = static_cast<IndexType>(adj[k]);
                queue.push(v);
            }
        }
    }
}

} // namespace pbat::graph

#endif // PBAT_GRAPH_BREADTHFIRSTSEARCH_H