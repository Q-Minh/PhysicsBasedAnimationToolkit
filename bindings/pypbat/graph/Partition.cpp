#include "Partition.h"

#include <pbat/graph/Partition.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace graph {

void BindPartition([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;

    using pbat::graph::PartitioningOptions;
    pyb::enum_<PartitioningOptions::EObjective>(m, "PartitioningObjective")
        .value("Default", PartitioningOptions::EObjective::Default)
        .value("MinEdgeCut", PartitioningOptions::EObjective::MinEdgeCut)
        .value("MinCommunicationVolume", PartitioningOptions::EObjective::MinCommunicationVolume)
        .export_values();

    pyb::enum_<PartitioningOptions::ECoarseningStrategy>(m, "PartitioningCoarseningStrategy")
        .value("Default", PartitioningOptions::ECoarseningStrategy::Default)
        .value("RandomMatching", PartitioningOptions::ECoarseningStrategy::RandomMatching)
        .value(
            "SortedHeavyEdgeMatching",
            PartitioningOptions::ECoarseningStrategy::SortedHeavyEdgeMatching)
        .export_values();

    pyb::enum_<PartitioningOptions::EInitialPartitioningStrategy>(m, "InitialPartitioningStrategy")
        .value("Default", PartitioningOptions::EInitialPartitioningStrategy::Default)
        .value(
            "GreedyBisectionGrowing",
            PartitioningOptions::EInitialPartitioningStrategy::GreedyBisectionGrowing)
        .value(
            "RandomBisectionAndRefinement",
            PartitioningOptions::EInitialPartitioningStrategy::RandomBisectionAndRefinement)
        .value(
            "EdgeCutSeparator",
            PartitioningOptions::EInitialPartitioningStrategy::EdgeCutSeparator)
        .value(
            "GreedyNodeBisectionGrowing",
            PartitioningOptions::EInitialPartitioningStrategy::GreedyNodeBisectionGrowing)
        .export_values();

    pyb::enum_<PartitioningOptions::ERefinementStrategy>(m, "PartitioningRefinementStrategy")
        .value("Default", PartitioningOptions::ERefinementStrategy::Default)
        .value("FiducciaMattheyses", PartitioningOptions::ERefinementStrategy::FiducciaMattheyses)
        .value(
            "GreedyCutAndVolumeRefinement",
            PartitioningOptions::ERefinementStrategy::GreedyCutAndVolumeRefinement)
        .value(
            "TwoSidedNodeFiducciaMattheyses",
            PartitioningOptions::ERefinementStrategy::TwoSidedNodeFiducciaMattheyses)
        .value(
            "OneSidedNodeFiducciaMattheyses",
            PartitioningOptions::ERefinementStrategy::OneSidedNodeFiducciaMattheyses)
        .export_values();

    PartitioningOptions opts{};
    m.def(
        "partition",
        [](IndexVectorX const& ptr,
           IndexVectorX const& adj,
           IndexVectorX const& wadj,
           Index nPartitions,
           PartitioningOptions::EObjective eObjective,
           PartitioningOptions::ECoarseningStrategy eCoarsen,
           PartitioningOptions::EInitialPartitioningStrategy eIP,
           PartitioningOptions::ERefinementStrategy eRefine,
           int nPartTrials,
           int nSeps,
           int nRIters,
           int seed,
           bool bMinDegree,
           bool b2Hop,
           bool bContig,
           bool bConnComp) {
            PartitioningOptions optsPass{};
            optsPass.eObjective                     = eObjective;
            optsPass.eCoarseningStrategy            = eCoarsen;
            optsPass.eInitialPartitioningStrategy   = eIP;
            optsPass.eRefinementStrategy            = eRefine;
            optsPass.nPartitioningTrials            = nPartTrials;
            optsPass.nSeparators                    = nSeps;
            optsPass.nRefinementIters               = nRIters;
            optsPass.rngSeed                        = seed;
            optsPass.bMinimizeSupernodalGraphDegree = bMinDegree;
            optsPass.bPerform2HopMatching           = b2Hop;
            optsPass.bEnforceContiguousPartitions   = bContig;
            optsPass.bIdentifyConnectedComponents   = bConnComp;
            return pbat::graph::Partition(ptr, adj, wadj, nPartitions, optsPass);
        },
        pyb::arg("ptr"),
        pyb::arg("adj"),
        pyb::arg("wgt"),
        pyb::arg("n_partitions"),
        pyb::arg("objective")          = opts.eObjective,
        pyb::arg("coarsening")         = opts.eCoarseningStrategy,
        pyb::arg("initializer")        = opts.eInitialPartitioningStrategy,
        pyb::arg("refinement")         = opts.eRefinementStrategy,
        pyb::arg("n_partition_trials") = opts.nPartitioningTrials,
        pyb::arg("n_separators")       = opts.nSeparators,
        pyb::arg("n_refinement_iters") = opts.nRefinementIters,
        pyb::arg("seed")               = opts.rngSeed,
        pyb::arg("minimize_degree")    = opts.bMinimizeSupernodalGraphDegree,
        pyb::arg("with_two_hop")       = opts.bPerform2HopMatching,
        pyb::arg("contiguous_parts")   = opts.bEnforceContiguousPartitions,
        pyb::arg("identify_conn_comp") = opts.bIdentifyConnectedComponents,
        "Partition a weighted directed graph (ptr, adj, wgt) given in sparse compressed format "
        "into n_partitions. Returns the |len(ptr)-1| partitioning map p such that p[v] yields the "
        "partition containing vertex v.");
}

} // namespace graph
} // namespace py
} // namespace pbat