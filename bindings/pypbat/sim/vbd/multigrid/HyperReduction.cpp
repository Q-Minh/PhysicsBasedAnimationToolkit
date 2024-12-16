#include "HyperReduction.h"

#include <pbat/sim/vbd/Data.h>
#include <pbat/sim/vbd/multigrid/HyperReduction.h>
#include <pybind11/eigen.h>

namespace pbat {
namespace py {
namespace sim {
namespace vbd {
namespace multigrid {

void BindHyperReduction(pybind11::module& m)
{
    namespace pyb = pybind11;
    using pbat::sim::vbd::Data;
    using pbat::sim::vbd::multigrid::HyperReduction;
    using namespace pbat::sim::vbd::multigrid::hypre;

    pyb::enum_<EClusteringStrategy>(m, "EClusteringStrategy")
        .value("Off", EClusteringStrategy::None)
        .value("Cluster", EClusteringStrategy::Cluster)
        .export_values();

    pyb::enum_<EElementSelectionStrategy>(m, "EElementSelectionStrategy")
        .value("All", EElementSelectionStrategy::All)
        .value("ClusterCenter", EElementSelectionStrategy::ClusterCenter)
        .value("MaxElasticEnergy", EElementSelectionStrategy::MaxElasticEnergy)
        .value("MinElasticEnergy", EElementSelectionStrategy::MinElasticEnergy)
        .value("MedianElasticEnergy", EElementSelectionStrategy::MedianElasticEnergy)
        .export_values();

    pyb::enum_<EVertexSelectionStrategy>(m, "EVertexSelectionStrategy")
        .value("All", EVertexSelectionStrategy::All)
        .value("ElementVertices", EVertexSelectionStrategy::ElementVertices)
        .export_values();

    pyb::enum_<EPotentialIntegrationStrategy>(m, "EPotentialIntegrationStrategy")
        .value("FineElementQuadWeights", EPotentialIntegrationStrategy::FineElementQuadWeights)
        .value("ClusterQuadWeightSum", EPotentialIntegrationStrategy::ClusterQuadWeightSum)
        .value("MatchPreStepElasticity", EPotentialIntegrationStrategy::MatchPreStepElasticity)
        .export_values();

    pyb::enum_<EKineticIntegrationStrategy>(m, "EKineticIntegrationStrategy")
        .value("FineVertexMasses", EKineticIntegrationStrategy::FineVertexMasses)
        .value("MatchTotalMass", EKineticIntegrationStrategy::MatchTotalMass)
        .export_values();

    pyb::class_<Strategies>(m, "HyperReductionStrategies")
        .def(
            pyb::init<
                EClusteringStrategy,
                EElementSelectionStrategy,
                EVertexSelectionStrategy,
                EPotentialIntegrationStrategy,
                EKineticIntegrationStrategy>(),
            pyb::arg("clustering_strategy")        = EClusteringStrategy::None,
            pyb::arg("element_selection_strategy") = EElementSelectionStrategy::All,
            pyb::arg("vertex_selection_strategy")  = EVertexSelectionStrategy::All,
            pyb::arg("potential_integration_strategy") =
                EPotentialIntegrationStrategy::FineElementQuadWeights,
            pyb::arg("kinetic_integration_strategy") =
                EKineticIntegrationStrategy::FineVertexMasses)
        .def_readwrite("clustering_strategy", &Strategies::eClustering)
        .def_readwrite("element_selection_strategy", &Strategies::eElementSelection)
        .def_readwrite("vertex_selection_strategy", &Strategies::eVertexSelection)
        .def_readwrite("potential_integration_strategy", &Strategies::ePotentialIntegration)
        .def_readwrite("kinetic_integration_strategy", &Strategies::eKineticIntegration);

    pyb::class_<HyperReduction>(m, "HyperReduction")
        .def(pyb::init<>())
        .def(
            pyb::init<Data const&, Index, Strategies>(),
            pyb::arg("data"),
            pyb::arg("n_target_active_elements"),
            pyb::arg("strategies") = Strategies{})
        .def_readwrite(
            "active_elements",
            &HyperReduction::bActiveE,
            "|#fine elements| boolean mask identifying active elements at this coarse level")
        .def_readwrite(
            "active_vertices",
            &HyperReduction::bActiveK,
            "|#fine vertices| boolean mask identifying active vertices at this coarse level")
        .def_readwrite(
            "cluster_ptr",
            &HyperReduction::clusterPtr,
            "|#clusters+1| indptr array of element clusters")
        .def_readwrite(
            "cluster_adj",
            &HyperReduction::clusterAdj,
            "|#fine elements| indices of clustered fine elements")
        .def_readwrite(
            "wgE",
            &HyperReduction::wgE,
            "|#fine elements| quadrature weights at this coarse level")
        .def_readwrite(
            "mK",
            &HyperReduction::mK,
            "|#fine vertices| lumped nodal masses at this coarse level")
        .def_readwrite("strategies", &HyperReduction::strategies, "Hyper reduction approach used");
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace py
} // namespace pbat