#ifndef PBAT_SIM_VBD_MULTIGRID_HYPER_REDUCTION_H
#define PBAT_SIM_VBD_MULTIGRID_HYPER_REDUCTION_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Data;

namespace multigrid {
namespace hypre {

enum class EClusteringStrategy { None, Cluster };
enum class EElementSelectionStrategy {
    All,
    ClusterCenter,
    MaxElasticEnergy,
    MinElasticEnergy,
    MedianElasticEnergy
};
enum class EVertexSelectionStrategy { All, ElementVertices };
enum class EPotentialIntegrationStrategy {
    FineElementQuadWeights,
    ClusterQuadWeightSum,
    MatchPreStepElasticity
};
enum class EKineticIntegrationStrategy { FineVertexMasses, MatchTotalMass };

struct Strategies
{
    Strategies(
        EClusteringStrategy eClustering             = EClusteringStrategy::None,
        EElementSelectionStrategy eElementSelection = EElementSelectionStrategy::All,
        EVertexSelectionStrategy eVertexSelection   = EVertexSelectionStrategy::All,
        EPotentialIntegrationStrategy ePotentialIntegration =
            EPotentialIntegrationStrategy::FineElementQuadWeights,
        EKineticIntegrationStrategy eKineticIntegration =
            EKineticIntegrationStrategy::FineVertexMasses);

    EClusteringStrategy eClustering;
    EElementSelectionStrategy eElementSelection;
    EVertexSelectionStrategy eVertexSelection;
    EPotentialIntegrationStrategy ePotentialIntegration;
    EKineticIntegrationStrategy eKineticIntegration;
};

} // namespace hypre

struct HyperReduction
{
    using BoolVectorX = Eigen::Vector<bool, Eigen::Dynamic>;

    HyperReduction() = default;

    HyperReduction(
        Data const& data,
        Index nTargetActiveElements,
        hypre::Strategies hyperReductionStrategies = hypre::Strategies{});

    void Update(Data const& data);

    BoolVectorX bActiveE; ///<
    BoolVectorX bActiveK; ///<

    VectorX wgE; ///<
    VectorX mK;  ///<

    IndexVectorX clusterPtr, clusterAdj; ///< Element clusters
    hypre::Strategies strategies;        ///< Hyper reduction approach
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_HYPER_REDUCTION_H