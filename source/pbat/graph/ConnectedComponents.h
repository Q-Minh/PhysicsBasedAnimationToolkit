#ifndef PBAT_GRAPH_CONNECTEDCOMPONENTS_H
#define PBAT_GRAPH_CONNECTEDCOMPONENTS_H

#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/graph/BreadthFirstSearch.h"
#include "pbat/graph/DepthFirstSearch.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/Core>
#include <cassert>

namespace pbat::graph {

/**
 * @brief Compute connected components of a graph using depth-first search
 * @tparam TIndex Index type used in the graph
 * @tparam TDerivedP Type of the pointer vector (adjacency list start indices for each vertex)
 * @tparam TDerivedAdj Type of the adjacency list (vector of vertex indices)
 * @param ptr Pointer to the start of each vertex's adjacency list
 * @param adj Adjacency list of the graph
 * @param components Output vector to store component labels for each vertex
 * @param dfs Depth-first search object
 * @return Number of connected components found in the graph
 * @pre `components[u] < 0` for all vertices `u` in the graph
 */
template <common::CIndex TIndex, class TDerivedP, class TDerivedAdj>
TIndex ConnectedComponents(
    Eigen::DenseBase<TDerivedP> const& ptr,
    Eigen::DenseBase<TDerivedAdj> const& adj,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic>> components,
    DepthFirstSearch<TIndex>& dfs)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.ConnectedComponents");
    Eigen::Index nVertices = ptr.size() - 1;
    assert(dfs.NumVertices() == nVertices);
    TIndex c{0};
    for (TIndex u = 0; u < nVertices; ++u)
    {
        if (components[u] < TIndex(0))
        {
            components[u] = c;
            dfs(ptr, adj, u, [&](TIndex v) { components[v] = c; });
            ++c;
        }
    }
    return c;
}

/**
 * @brief Compute connected components of a graph using breadth-first search
 * @tparam TIndex Index type used in the graph
 * @tparam TDerivedP Type of the pointer vector (adjacency list start indices for each vertex)
 * @tparam TDerivedAdj Type of the adjacency list (vector of vertex indices)
 * @param ptr Pointer to the start of each vertex's adjacency list
 * @param adj Adjacency list of the graph
 * @param components Output vector to store component labels for each vertex
 * @param bfs Breadth-first search object
 * @return Number of connected components found in the graph
 * @pre `components[u] < 0` for all vertices `u` in the graph
 */
template <common::CIndex TIndex, class TDerivedP, class TDerivedAdj>
TIndex ConnectedComponents(
    Eigen::DenseBase<TDerivedP> const& ptr,
    Eigen::DenseBase<TDerivedAdj> const& adj,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic>> components,
    BreadthFirstSearch<TIndex>& bfs)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.ConnectedComponents");
    Eigen::Index nVertices = ptr.size() - 1;
    assert(bfs.NumVertices() == nVertices);
    TIndex c{0};
    for (TIndex u = 0; u < nVertices; ++u)
    {
        if (components[u] < TIndex(0))
        {
            components[u] = c;
            bfs(ptr, adj, u, [&](TIndex v) { components[v] = c; });
            ++c;
        }
    }
    return c;
}

} // namespace pbat::graph

#endif // PBAT_GRAPH_CONNECTEDCOMPONENTS_H