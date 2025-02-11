/**
 * @file Partition.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Graph partitioning API
 * @date 2025-02-10
 *
 * @copyright Copyright (c) 2025
 * @ingroup graph
 */

#ifndef PBAT_GRAPH_PARTITION_H
#define PBAT_GRAPH_PARTITION_H

#include "pbat/Aliases.h"

namespace pbat {
namespace graph {

/**
 * @brief Options for graph partitioning
 *
 * Refer to [METIS](https://github.com/KarypisLab/METIS) manual for more details on each option.
 *
 * @ingroup graph
 */
struct PartitioningOptions
{
    /**
     * @brief Objective function for partitioning
     */
    enum class EObjective {
        Default,                       ///< Default objective function
        MinEdgeCut,                    ///< Minimize edge cut
        MinCommunicationVolume,        ///< Minimize communication volume
    } eObjective{EObjective::Default}; ///< Objective function for partitioning
    /**
     * @brief Coarsening strategy
     */
    enum class ECoarseningStrategy {
        Default,                                         ///< Default coarsening strategy
        RandomMatching,                                  ///< Random matching
        SortedHeavyEdgeMatching                          ///< Sorted heavy edge matching
    } eCoarseningStrategy{ECoarseningStrategy::Default}; ///< Coarsening strategy
    /**
     * @brief Initial partitioning strategy
     */
    enum class EInitialPartitioningStrategy {
        Default,                      ///< Default initial partitioning strategy
        GreedyBisectionGrowing,       ///< Greedy bisection growing
        RandomBisectionAndRefinement, ///< Random bisection and refinement
        EdgeCutSeparator,             ///< Edge cut separator
        GreedyNodeBisectionGrowing    ///< Greedy node bisection growing
    } eInitialPartitioningStrategy{
        EInitialPartitioningStrategy::Default}; ///< Initial partitioning strategy
    /**
     * @brief Refinement strategy
     */
    enum class ERefinementStrategy {
        Default,                                         ///< Default refinement strategy
        FiducciaMattheyses,                              ///< Fiduccia-Mattheyses
        GreedyCutAndVolumeRefinement,                    ///< Greedy cut and volume refinement
        TwoSidedNodeFiducciaMattheyses,                  ///< Two-sided node Fiduccia-Matthe
        OneSidedNodeFiducciaMattheyses                   ///< One-sided node Fiduccia-Mattheyses
    } eRefinementStrategy{ERefinementStrategy::Default}; ///< Refinement strategy
    int nPartitioningTrials{1};                          ///< Number of partitioning trials
    int nSeparators{1};                                  ///< Number of separators
    int nRefinementIters{10};                            ///< Number of refinement iterations
    int rngSeed{0};                                      ///< Random number generator seed
    bool bMinimizeSupernodalGraphDegree{false};          ///< Minimize supernodal graph degree
    bool bPerform2HopMatching{true};                     ///< Perform 2-hop matching
    bool bEnforceContiguousPartitions{false};            ///< Enforce contiguous partitions
    bool bIdentifyConnectedComponents{false};            ///< Identify connected components
};

/**
 * @brief Partition input graph
 *
 * @param ptr Offset pointers of adjacency list
 * @param adj Indices of adjacency list
 * @param wadj Edge weights of adjacency list
 * @param nPartitions Number of desired partitions
 * @param opts Partitioning options
 * @return IndexVectorX |# vertices| array p of partition indices (i.e. p[i] = partition of vertex i)
 * @ingroup graph
 */
IndexVectorX Partition(
    Eigen::Ref<IndexVectorX const> const& ptr,
    Eigen::Ref<IndexVectorX const> const& adj,
    Eigen::Ref<IndexVectorX const> const& wadj,
    Index nPartitions,
    PartitioningOptions opts = PartitioningOptions{});

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_PARTITION_H