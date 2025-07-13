#ifndef PBAT_GRAPH_DEPTH_FIRST_SEARCH_H
#define PBAT_GRAPH_DEPTH_FIRST_SEARCH_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/profiling/Profiling.h"

#include <stack>
#include <vector>

namespace pbat::graph {

template <common::CIndex TIndex = Index>
struct DepthFirstSearch
{
    using IndexType = TIndex; ///< Index type used in the depth-first search
    using StackType = std::stack<IndexType, std::vector<IndexType>>; ///< Stack type used for DFS

    DepthFirstSearch() = default;
    /**
     * @brief Construct a new Depth First Search object
     * @param n Number of vertices in the graph
     */
    DepthFirstSearch(Eigen::Index n);
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
    StackType stack;                             ///< DFS search stack
};

template <common::CIndex TIndex>
inline DepthFirstSearch<TIndex>::DepthFirstSearch(Eigen::Index n)
{
    Reserve(n);
}

template <common::CIndex TIndex>
inline void DepthFirstSearch<TIndex>::Reserve(Eigen::Index n)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DepthFirstSearch.Reserve");
    visited.resize(n);
    std::vector<IndexType> memory{};
    memory.reserve(static_cast<std::size_t>(n));
    stack = StackType(std::move(memory));
}

template <common::CIndex TIndex>
template <class FVisit, class TDerivedP, class TDerivedAdj>
inline void DepthFirstSearch<TIndex>::operator()(
    Eigen::DenseBase<TDerivedP> const& ptr,
    Eigen::DenseBase<TDerivedAdj> const& adj,
    TIndex start,
    FVisit fVisit)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.DepthFirstSearch");
    visited.setConstant(false);
    stack.push(start);
    while (not stack.empty())
    {
        IndexType u = stack.top();
        stack.pop();
        if (not visited[u])
        {
            fVisit(u);
            visited[u]  = true;
            auto kBegin = ptr[u];
            auto kEnd   = ptr[u + 1];
            for (auto k = kBegin; k < kEnd; ++k)
            {
                IndexType v = static_cast<IndexType>(adj[k]);
                stack.push(v);
            }
        }
    }
}

} // namespace pbat::graph

#endif // PBAT_GRAPH_DEPTH_FIRST_SEARCH_H