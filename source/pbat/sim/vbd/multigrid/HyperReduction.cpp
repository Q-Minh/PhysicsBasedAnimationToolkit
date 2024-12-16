#include "HyperReduction.h"

#include "pbat/common/ArgSort.h"
#include "pbat/common/Indexing.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/graph/Partition.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Data.h"

#include <ranges>
#include <utility>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

namespace hypre {

Strategies::Strategies(
    EClusteringStrategy eClusteringIn,
    EElementSelectionStrategy eElementSelectionIn,
    EVertexSelectionStrategy eVertexSelectionIn,
    EPotentialIntegrationStrategy ePotentialIntegrationIn,
    EKineticIntegrationStrategy eKineticIntegrationIn)
    : eClustering(eClusteringIn),
      eElementSelection(eElementSelectionIn),
      eVertexSelection(eVertexSelectionIn),
      ePotentialIntegration(ePotentialIntegrationIn),
      eKineticIntegration(eKineticIntegrationIn)
{
}

} // namespace hypre

HyperReduction::HyperReduction(
    Data const& data,
    Index nTargetActiveElements,
    hypre::Strategies strategiesIn)
    : bActiveE(BoolVectorX::Constant(data.mesh.E.cols(), true)),
      bActiveK(BoolVectorX::Constant(data.mesh.X.cols(), true)),
      wgE(data.wg),
      mK(data.m),
      clusterPtr(),
      clusterAdj(),
      strategies(strategiesIn)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Level.HyperReduce");

    if (strategies.eClustering == hypre::EClusteringStrategy::Cluster)
    {
        auto const nFineElements = data.mesh.E.cols();
        auto const nFineVertices = data.mesh.X.cols();

        // Choose coarse level compute budget as 3x its elements, but not surpassing the number of
        // fine elements. Compute the graph of face-adjacent tetrahedra on the fine mesh
        auto Gdual = graph::MeshDualGraph(data.mesh.E, nFineVertices);
        Gdual.prune([]([[maybe_unused]] Index row, [[maybe_unused]] Index col, Index value) {
            bool const bIsFaceAdjacency = value == 3;
            return bIsFaceAdjacency;
        });
        auto Gdualptr = graph::AdjacencyMatrixPrefix(Gdual);
        auto Gdualadj = graph::AdjacencyMatrixIndices(Gdual);
        // Partition the fine mesh using scale-invariant geometric distance between element
        // barycenters as the weight function.
        // TODO: Use smarter weight function that exhibits shape and material awareness.
        MatrixX XEbary = data.mesh.QuadraturePoints<1>();
        Scalar const scale =
            (data.mesh.X.rowwise().maxCoeff() - data.mesh.X.rowwise().minCoeff()).norm();
        Scalar decimalPrecision = 1e4;
        IndexVectorX Wdual(Gdual.nonZeros());
        graph::ForEachEdge(Gdualptr, Gdualadj, [&](Index ei, Index ej, Index k) {
            Scalar dij = (XEbary.col(ei) - XEbary.col(ej)).norm() / scale;
            Wdual(k)   = static_cast<Index>(dij * decimalPrecision);
        });
        IndexVectorX clusters = graph::Partition(Gdualptr, Gdualadj, Wdual, nTargetActiveElements);
        std::tie(clusterPtr, clusterAdj) = graph::MapToAdjacency(clusters);
        // Compute the coarse level quadrature weights of active fine elements as the sum of cluster
        // element weights.
        VectorX clusterQuadWeights = VectorX::Zero(nTargetActiveElements);
        graph::ForEachEdge(clusterPtr, clusterAdj, [&](Index c, Index e, [[maybe_unused]] Index k) {
            clusterQuadWeights(c) += data.wg(e);
        });
        if (strategies.eElementSelection == hypre::EElementSelectionStrategy::ClusterCenter)
        {
            bActiveE.setConstant(nFineElements, false);
            bActiveK.setConstant(nFineVertices, false);
            // We use the previously computed weighting scheme to compute "shape-aware" cluster
            // geometric centers
            MatrixX clusterCenters = MatrixX::Zero(3, nTargetActiveElements);
            graph::ForEachEdge(
                clusterPtr,
                clusterAdj,
                [&](Index c, Index e, [[maybe_unused]] Index k) {
                    clusterCenters.col(c) += data.wg(e) * XEbary.col(e) / clusterQuadWeights(c);
                });
            // Find active elements as those closest to cluster centers.
            VectorX clusterMinDistToCenter =
                VectorX::Constant(nTargetActiveElements, std::numeric_limits<Scalar>::max());
            IndexVectorX activeElements(nTargetActiveElements);
            graph::ForEachEdge(
                clusterPtr,
                clusterAdj,
                [&](Index c, Index e, [[maybe_unused]] Index k) {
                    Scalar d = (XEbary.col(e) - clusterCenters.col(c)).squaredNorm();
                    if (d >= clusterMinDistToCenter(c))
                        return;
                    activeElements(c)         = e;
                    clusterMinDistToCenter(c) = d;
                });
            bActiveE(activeElements).setConstant(true);
        }
        if (strategies.eVertexSelection == hypre::EVertexSelectionStrategy::ElementVertices)
        {
            IndexVectorX activeElements =
                common::Filter(0, nFineElements, [&](Index ef) { return bActiveE(ef); });
            // Make element vertices active
            IndexVectorX activeVertices =
                data.mesh.E(Eigen::placeholders::all, activeElements).reshaped();
            std::sort(activeVertices.begin(), activeVertices.end());
            auto const nActiveVertices = std::distance(
                activeVertices.begin(),
                std::unique(activeVertices.begin(), activeVertices.end()));
            activeVertices.conservativeResize(nActiveVertices);
            bActiveK(activeVertices).setConstant(true);
        }
        if (strategies.ePotentialIntegration ==
            hypre::EPotentialIntegrationStrategy::ClusterQuadWeightSum)
        {
            IndexVectorX activeElements =
                common::Filter(0, nFineElements, [&](Index ef) { return bActiveE(ef); });
            // Compute coarse level quadrature weights
            wgE.setZero(nFineElements);
            wgE(activeElements) = clusterQuadWeights;
        }
        if (strategies.eKineticIntegration == hypre::EKineticIntegrationStrategy::MatchTotalMass)
        {
            IndexVectorX activeVertices =
                common::Filter(0, nFineVertices, [&](Index vf) { return bActiveK(vf); });
            // Compute coarse level lumped masses by scaling s.t. total mass is matched,
            // i.e. \alpha * \sum Mcoarse(i) = Mtotal -> \alpha = Mtotal / \sum Mcoarse(i)
            Scalar alpha = data.m.sum() / data.m(activeVertices).sum();
            mK.setZero(nFineVertices);
            mK(activeVertices) = alpha * data.m(activeVertices);
        }

        // TODO:
        // We might want to make element selection this adaptive per timestep.
        // This could be in the form of keeping the clusters pre-computed, but changing the
        // active element in each cluster at the beginning of each time step based on strain
        // rate.

        // TODO:
        // Find smarter way to compute coarse quadrature weights.
        // This could potentially be an adaptive, per-timestep method
        // that computes a quadrature weight which, when multiplying
        // its corresponding active element's elasticity, would yield
        // the sum of its cluster's elements' elastic energies.

        // TODO:
        // Find smart way to reduce quadrature for Dirichlet boundary conditions and
        // contact constraints. This could be in the form of selecting only a fixed-size
        // subset of contacts with the deepest penetrations, while the Dirichlet reduction
        // should be based on shape preservation and can be precomputed.
    }
}

void HyperReduction::Update(Data const& data)
{
    bool bActiveElementSetChanged{false};
    if (strategies.eElementSelection == hypre::EElementSelectionStrategy::MaxElasticEnergy or
        strategies.eElementSelection == hypre::EElementSelectionStrategy::MinElasticEnergy or
        strategies.eElementSelection == hypre::EElementSelectionStrategy::MedianElasticEnergy)
    {
        bActiveE.setConstant(false);
        Index nClusters = clusterPtr.size() - 1;
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto kBegin          = clusterPtr(c);
            auto kEnd            = clusterPtr(c + 1);
            auto clusterElements = clusterAdj(Eigen::seqN(kBegin, kEnd - kBegin));
            Index kActive{0};
            if (strategies.eElementSelection == hypre::EElementSelectionStrategy::MaxElasticEnergy)
            {
                data.psiE(clusterElements).maxCoeff(&kActive);
            }
            if (strategies.eElementSelection == hypre::EElementSelectionStrategy::MinElasticEnergy)
            {
                data.psiE(clusterElements).minCoeff(&kActive);
            }
            if (strategies.eElementSelection ==
                hypre::EElementSelectionStrategy::MedianElasticEnergy)
            {
                VectorX psiE       = data.psiE(clusterElements);
                IndexVectorX order = common::ArgSort(psiE.size(), [&](Index i, Index j) {
                    return psiE(i) < psiE(j);
                });
                kActive            = order(order.size() / 2);
            }
            Index eActive     = clusterElements(kActive);
            bActiveE(eActive) = true;
        });
        bActiveElementSetChanged = true;
    }
    if (strategies.ePotentialIntegration ==
        hypre::EPotentialIntegrationStrategy::MatchPreStepElasticity)
    {
        auto nFineElements = data.mesh.E.cols();
        IndexVectorX activeElements =
            common::Filter(0, nFineElements, [&](Index e) { return bActiveE(e); });
        // For each active element EA, we want to find an appropriate quadrature weight wEA
        // such that wgEA * psi(EA) = \sum_EC wge * psi(EC),
        // where EC are elements belonging to the same cluster as EA, and wge their
        // corresponding quadrature weight.
        // Thus, wgEA = \sum_EC wge * psi(EC) / psi(EA)
        wgE.setZero();
        Index nClusters = clusterPtr.size() - 1;
        tbb::parallel_for(Index(0), nClusters, [&](Index c) {
            auto kBegin          = clusterPtr(c);
            auto kEnd            = clusterPtr(c + 1);
            auto clusterElements = clusterAdj(Eigen::seqN(kBegin, kEnd - kBegin));
            Index eActive        = activeElements(c);
            auto wgEC            = data.wg(clusterElements);
            auto psiEC           = data.psiE(clusterElements);
            wgE(eActive)         = wgEC.dot(psiEC) / data.psiE(eActive);
        });
    }
    if (bActiveElementSetChanged)
    {
        if (strategies.eVertexSelection == hypre::EVertexSelectionStrategy::ElementVertices)
        {
            bActiveK.setConstant(false);
            IndexVectorX activeElements =
                common::Filter(0, bActiveE.size(), [&](Index e) { return bActiveE(e); });
            // Make element vertices active
            IndexVectorX activeVertices =
                data.mesh.E(Eigen::placeholders::all, activeElements).reshaped();
            bActiveK(activeVertices).setConstant(true);
        }
        if (strategies.eKineticIntegration == hypre::EKineticIntegrationStrategy::MatchTotalMass)
        {
            IndexVectorX activeVertices =
                common::Filter(0, bActiveK.size(), [&](Index v) { return bActiveK(v); });
            auto nFineElements = data.mesh.E.cols();
            IndexVectorX activeElements =
                common::Filter(0, nFineElements, [&](Index e) { return bActiveE(e); });
            mK.setZero();
            Scalar alpha       = data.m.sum() / data.m(activeVertices).sum();
            mK(activeVertices) = alpha * data.m(activeVertices);
        }
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat
