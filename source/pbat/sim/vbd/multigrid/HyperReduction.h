#ifndef PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H
#define PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H

#include "pbat/Aliases.h"

#include <vector>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Hierarchy;

struct HyperReduction
{
    static constexpr auto kPolynomialOrder = 1;
    static constexpr auto kDims            = 3;

    HyperReduction() = default;
    /**
     * @brief
     *
     * @param hierarchy
     * @param clusterSize
     */
    HyperReduction(Hierarchy const& hierarchy, Index clusterSize = 5);
    /**
     * @brief
     *
     * @param hierarchy
     * @param clusterSize
     */
    void Construct(Hierarchy const& hierarchy, Index clusterSize = 5);
    /**
     * @brief
     *
     * @param nElements
     * @param nLevels
     */
    void AllocateWorkspace(Index nElements, std::size_t nLevels);
    /**
     * @brief Construct the hierarchical clustering of mesh elements
     *
     * @param hierarchy
     * @param clusterSize
     */
    void ConstructHierarchicalClustering(Hierarchy const& hierarchy, Index clusterSize);
    /**
     * @brief
     *
     * @param hierarchy
     */
    void SelectClusterRepresentatives(Hierarchy const& hierarchy);
    /**
     * @brief
     *
     * @param hierarchy
     */
    void PrecomputeInversePolynomialMatrices(Hierarchy const& hierarchy);

    /**
     * @brief Hierarchical clustering of mesh elements
     */
    std::vector<IndexVectorX> Cptr;
    std::vector<IndexVectorX> Cadj;

    std::vector<IndexVectorX> eC; ///< |#levels| list of |#clusters| arrays of representative
                                  ///< elements in each cluster of the corresponding level
    std::vector<MatrixX>
        ApInvC; ///< |#levels| list of 4x|4*#clusters| of A_p^{-1} matrices, such that A_p's
                ///< coefficients are \int_{\Omega^c} P_i(X) P_j(X) dx, where \Omega^c is cluster
                ///< c's domain, and P_k(X) is the k^{th} polynomial basis.
    std::vector<VectorX>
        Ep;             ///< |#levels+1| linear polynomial errors at fine elements and at each level
    Scalar EpMax{1e-6}; ///< Maximum allowable linear polynomial error in any cluster
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H
