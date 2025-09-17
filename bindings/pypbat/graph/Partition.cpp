#include "Partition.h"

#include <pbat/graph/Partition.h>
#include <nanobind/eigen/dense.h>

namespace pbat {
namespace py {
namespace graph {

void BindPartition([[maybe_unused]] nanobind::module_& m)
{
    namespace nb = nanobind;

    using pbat::graph::PartitioningOptions;
    nb::enum_<PartitioningOptions::EObjective>(m, "PartitioningObjective")
        .value("Default", PartitioningOptions::EObjective::Default)
        .value("MinEdgeCut", PartitioningOptions::EObjective::MinEdgeCut)
        .value("MinCommunicationVolume", PartitioningOptions::EObjective::MinCommunicationVolume)
        .export_values();

    nb::enum_<PartitioningOptions::ECoarseningStrategy>(m, "PartitioningCoarseningStrategy")
        .value("Default", PartitioningOptions::ECoarseningStrategy::Default)
        .value("RandomMatching", PartitioningOptions::ECoarseningStrategy::RandomMatching)
        .value(
            "SortedHeavyEdgeMatching",
            PartitioningOptions::ECoarseningStrategy::SortedHeavyEdgeMatching)
        .export_values();

    nb::enum_<PartitioningOptions::EInitialPartitioningStrategy>(m, "InitialPartitioningStrategy")
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

    nb::enum_<PartitioningOptions::ERefinementStrategy>(m, "PartitioningRefinementStrategy")
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
        nb::arg("ptr"),
        nb::arg("adj"),
        nb::arg("wgt"),
        nb::arg("n_partitions"),
        nb::arg("objective")          = opts.eObjective,
        nb::arg("coarsening")         = opts.eCoarseningStrategy,
        nb::arg("initializer")        = opts.eInitialPartitioningStrategy,
        nb::arg("refinement")         = opts.eRefinementStrategy,
        nb::arg("n_partition_trials") = opts.nPartitioningTrials,
        nb::arg("n_separators")       = opts.nSeparators,
        nb::arg("n_refinement_iters") = opts.nRefinementIters,
        nb::arg("seed")               = opts.rngSeed,
        nb::arg("minimize_degree")    = opts.bMinimizeSupernodalGraphDegree,
        nb::arg("with_two_hop")       = opts.bPerform2HopMatching,
        nb::arg("contiguous_parts")   = opts.bEnforceContiguousPartitions,
        nb::arg("identify_conn_comp") = opts.bIdentifyConnectedComponents,
        "Partition a weighted directed graph (ptr, adj, wgt) given in sparse compressed format "
        "into n_partitions. Returns the |len(ptr)-1| partitioning map p such that p[v] yields the "
        "partition containing vertex v.");
}

} // namespace graph
} // namespace py
} // namespace pbat