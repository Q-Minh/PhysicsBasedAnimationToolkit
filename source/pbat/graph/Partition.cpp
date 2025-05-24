#include "Partition.h"

#include "pbat/common/Eigen.h"
#include "pbat/profiling/Profiling.h"

#ifdef PBAT_USE_METIS
    #include <metis.h>
#endif // PBAT_USE_METIS

#include <exception>
#include <string>
#include <type_traits>

namespace pbat {
namespace graph {

IndexVectorX Partition(
    [[maybe_unused]] Eigen::Ref<IndexVectorX const> const& ptr,
    [[maybe_unused]] Eigen::Ref<IndexVectorX const> const& adj,
    [[maybe_unused]] Eigen::Ref<IndexVectorX const> const& wadj,
    [[maybe_unused]] Index nPartitions,
    [[maybe_unused]] PartitioningOptions opts)
{
#ifdef PBAT_USE_METIS
    PBAT_PROFILE_NAMED_SCOPE("pbat.graph.Partition");
    // Use Metis to partition the graph
    auto xadj       = ptr.cast<idx_t>().eval();
    auto adjncy     = adj.cast<idx_t>().eval();
    auto adjwgt     = wadj.cast<idx_t>().eval();
    idx_t nVertices = static_cast<idx_t>(ptr.size() - 1);
    idx_t nBalancingConstraints(1);
    idx_t nparts = static_cast<idx_t>(nPartitions);
    // Set METIS partitioner options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE]     = METIS_PTYPE_KWAY;
    options[METIS_OPTION_NUMBERING] = 0;
    switch (opts.eObjective)
    {
        case PartitioningOptions::EObjective::MinEdgeCut:
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
            break;
        case PartitioningOptions::EObjective::MinCommunicationVolume:
            options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
            break;
        case PartitioningOptions::EObjective::Default: [[fallthrough]];
        default: break;
    }
    switch (opts.eCoarseningStrategy)
    {
        case PartitioningOptions::ECoarseningStrategy::RandomMatching:
            options[METIS_OPTION_CTYPE] = METIS_CTYPE_RM;
            break;
        case PartitioningOptions::ECoarseningStrategy::SortedHeavyEdgeMatching:
            options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;
            break;
        case PartitioningOptions::ECoarseningStrategy::Default: [[fallthrough]];
        default: break;
    }
    switch (opts.eInitialPartitioningStrategy)
    {
        case PartitioningOptions::EInitialPartitioningStrategy::GreedyBisectionGrowing:
            options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;
            break;
        case PartitioningOptions::EInitialPartitioningStrategy::RandomBisectionAndRefinement:
            options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_RANDOM;
            break;
        case PartitioningOptions::EInitialPartitioningStrategy::EdgeCutSeparator:
            options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_EDGE;
            break;
        case PartitioningOptions::EInitialPartitioningStrategy::GreedyNodeBisectionGrowing:
            options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_NODE;
            break;
        case PartitioningOptions::EInitialPartitioningStrategy::Default: [[fallthrough]];
        default: break;
    }
    switch (opts.eRefinementStrategy)
    {
        case PartitioningOptions::ERefinementStrategy::FiducciaMattheyses:
            options[METIS_OPTION_RTYPE] = METIS_RTYPE_FM;
            break;
        case PartitioningOptions::ERefinementStrategy::GreedyCutAndVolumeRefinement:
            options[METIS_OPTION_RTYPE] = METIS_RTYPE_GREEDY;
            break;
        case PartitioningOptions::ERefinementStrategy::TwoSidedNodeFiducciaMattheyses:
            options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP2SIDED;
            break;
        case PartitioningOptions::ERefinementStrategy::OneSidedNodeFiducciaMattheyses:
            options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;
            break;
        case PartitioningOptions::ERefinementStrategy::Default: [[fallthrough]];
        default: break;
    }
    options[METIS_OPTION_NCUTS]   = static_cast<idx_t>(opts.nPartitioningTrials);
    options[METIS_OPTION_NSEPS]   = static_cast<idx_t>(opts.nSeparators);
    options[METIS_OPTION_NITER]   = static_cast<idx_t>(opts.nRefinementIters);
    options[METIS_OPTION_SEED]    = static_cast<idx_t>(opts.rngSeed);
    options[METIS_OPTION_MINCONN] = opts.bMinimizeSupernodalGraphDegree ? idx_t(1) : idx_t(0);
    options[METIS_OPTION_NO2HOP]  = opts.bPerform2HopMatching ? idx_t(0) : idx_t(1);
    options[METIS_OPTION_CONTIG]  = opts.bEnforceContiguousPartitions ? idx_t(1) : idx_t(0);
    options[METIS_OPTION_CCORDER] = opts.bIdentifyConnectedComponents ? idx_t(1) : idx_t(0);
    // Invoke partitioning implementation
    idx_t objval(-1);
    std::vector<idx_t> part(static_cast<std::size_t>(nVertices), idx_t(-1));
    int const ec = METIS_PartGraphKway(
        &nVertices,
        &nBalancingConstraints,
        xadj.data(),
        adjncy.data(),
        NULL,
        NULL,
        adjwgt.data(),
        &nparts,
        NULL,
        NULL,
        options,
        &objval,
        part.data());
    if (ec != METIS_OK)
    {
        std::string const what =
            (ec == METIS_ERROR_INPUT)  ? "Invalid input to METIS" :
            (ec == METIS_ERROR_MEMORY) ? "METIS could not allocate required memory" :
                                         "Unknown error from METIS";
        throw std::invalid_argument(what);
    }
    auto pbatPart = common::ToEigen(part).cast<Index>();
    return pbatPart;
#else
    throw std::runtime_error(
        "Attempted to use graph partitioning API, but library was not built with METIS.");
#endif // PBAT_USE_METIS
}

} // namespace graph
} // namespace pbat