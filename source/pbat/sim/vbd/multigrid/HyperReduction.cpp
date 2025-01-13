#include "HyperReduction.h"

#include "pbat/common/Indexing.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/graph/Partition.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"
#include "pbat/sim/vbd/multigrid/Hierarchy.h"

#include <ranges>
#include <tbb/parallel_for.h>
#include <utility>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

HyperReduction::HyperReduction(Hierarchy const& hierarchy, Index clusterSize)
    : Cptr(hierarchy.levels.size()),
      Cadj(hierarchy.levels.size()),
      Ep(hierarchy.levels.size() + 1),
      EpMax(0)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.HyperReduction.Construct");

    IndexVectorX Gptr, Gadj, Gwts;
    std::tie(Gptr, Gadj, Gwts) = graph::MatrixToWeightedAdjacency(graph::MeshDualGraph(
        hierarchy.data.E,
        hierarchy.data.X.cols(),
        graph::EMeshDualGraphOptions::FaceAdjacent));
    std::size_t nLevels        = hierarchy.levels.size();
    for (std::size_t l = 0; l < nLevels; l++)
    {
        // Reduce the graph via partitioning/clustering
        auto nGraphNodes = Gptr.size() - 1;
        auto nPartitions = nGraphNodes / clusterSize;
        graph::PartitioningOptions opts{};
        opts.eCoarseningStrategy =
            graph::PartitioningOptions::ECoarseningStrategy::SortedHeavyEdgeMatching;
        opts.rngSeed                        = 0;
        opts.bMinimizeSupernodalGraphDegree = true;
        opts.bEnforceContiguousPartitions   = true;
        opts.bIdentifyConnectedComponents   = true;
        auto clustering                     = graph::Partition(Gptr, Gadj, Gwts, nPartitions, opts);
        // Store the l^{th} level clustering
        std::tie(Cptr[l], Cadj[l]) = graph::MapToAdjacency(clustering);
        // Compute the supernodal graph (i.e. graph of clusters) as next graph to partition
        auto Gsizes   = Gptr(Eigen::seqN(1, nGraphNodes)) - Gptr(Eigen::seqN(0, nGraphNodes));
        auto iota     = IndexVectorX::LinSpaced(clustering.size(), 0, clustering.size() - 1);
        auto SGu      = clustering(common::Repeat(iota, Gsizes));
        auto SGv      = clustering(Gadj);
        auto edgeView = std::views::iota(0, SGu.size()) | std::views::transform([&](auto i) {
                            return graph::WeightedEdge(SGu(i), SGv(i), Gwts(i));
                        }) |
                        std::views::common;
        // NOTE:
        // Unfortunately, Eigen does not support std::ranges iterators, because it expects iterators
        // to have operator->. We have to make a copy into a std::vector to use Eigen's sparse
        // matrix.
        std::vector<graph::WeightedEdge<Index, Index>> edges(edgeView.begin(), edgeView.end());
        std::tie(Gptr, Gadj, Gwts) = graph::MatrixToWeightedAdjacency(
            graph::AdjacencyMatrixFromEdges(edges.begin(), edges.end()));
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat
