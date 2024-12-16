#ifndef PBAT_GRAPH_PARTITION_H
#define PBAT_GRAPH_PARTITION_H

#include "pbat/Aliases.h"

namespace pbat {
namespace graph {

struct PartitioningOptions
{
    enum class EObjective {
        Default,
        MinEdgeCut,
        MinCommunicationVolume,
    } eObjective{EObjective::Default};
    enum class ECoarseningStrategy {
        Default,
        RandomMatching,
        SortedHeavyEdgeMatching
    } eCoarseningStrategy{ECoarseningStrategy::Default};
    enum class EInitialPartitioningStrategy {
        Default,
        GreedyBisectionGrowing,
        RandomBisectionAndRefinement,
        EdgeCutSeparator,
        GreedyNodeBisectionGrowing
    } eInitialPartitioningStrategy{EInitialPartitioningStrategy::Default};
    enum class ERefinementStrategy {
        Default,
        FiducciaMattheyses,
        GreedyCutAndVolumeRefinement,
        TwoSidedNodeFiducciaMattheyses,
        OneSidedNodeFiducciaMattheyses
    } eRefinementStrategy{ERefinementStrategy::Default};
    int nPartitioningTrials{1};
    int nSeparators{1};
    int nRefinementIters{10};
    int rngSeed{0};
    bool bMinimizeSupernodalGraphDegree{false};
    bool bPerform2HopMatching{true};
    bool bEnforceContiguousPartitions{false};
    bool bIdentifyConnectedComponents{false};
};

IndexVectorX Partition(
    IndexVectorX const& ptr,
    IndexVectorX const& adj,
    IndexVectorX const& wadj,
    Index nPartitions,
    PartitioningOptions opts = PartitioningOptions{});

} // namespace graph
} // namespace pbat

#endif // PBAT_GRAPH_PARTITION_H