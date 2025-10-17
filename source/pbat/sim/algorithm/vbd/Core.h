/**
 * @file Core.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Core VBD API.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_CORE_H
#define PBAT_SIM_ALGORITHM_VBD_CORE_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"

#include <Eigen/Core>

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Initialization strategies for the VBD time step minimization
 */
enum class EInitializationStrategy {
    Position,             ///< \f$ x_0 = x(t) \f$
    Inertia,              ///< \f$ x_0 = x(t) + h v(t) \f$
    KineticEnergyMinimum, ///< \f$ x_0 = x(t) + h v(t) + h^2 M^{-1} f_\text{ext} \f$
    AdaptiveVbd,          ///< Adaptive VBD initialization strategy
    AdaptivePbat          ///< Adaptive PBAT initialization strategy
};

/**
 * @brief VBD simulation configuration
 */
struct Params
{
  public:
    /**
     * @brief Vertex-element adjacency graph
     * @param _GVGp Vertex graph pointer
     * @param _GVGe Vertex graph element indices
     * @param _GVGilocal Vertex graph local vertex indices
     * @return Reference to this
     */
    PBAT_API Params& WithVertexElementAdjacencyGraph(
        Eigen::Ref<IndexVectorX const> const& _GVGp,
        Eigen::Ref<IndexVectorX const> const& _GVGe,
        Eigen::Ref<IndexVectorX const> const& _GVGilocal);
    /**
     * @brief Vertex colors
     * @param _colors Vertex colors
     * @return Reference to this
     */
    PBAT_API Params& WithVertexColors(Eigen::Ref<IndexVectorX const> const& _colors);
    /**
     * @brief BCD optimization initialization strategy
     * @param _strategy Initialization strategy
     * @return Reference to this
     */
    PBAT_API Params& WithInitializationStrategy(EInitializationStrategy _strategy);
    /**
     * @brief Numerical zero for hessian pseudo-singularity check
     * @param zero Numerical zero
     * @return Reference to this
     */
    PBAT_API Params& WithHessianDeterminantZeroUnder(Scalar zero);
    /**
     * @brief Construct the simulation data
     * @param bValidate Throw on detected ill-formed inputs
     * @return Reference to this
     */
    PBAT_API Params& Construct(bool bValidate = true);

  public:
    // Vertex-element adjacency graph
    IndexVectorX GVGp;      ///< `|# verts+1|` prefixes into GVGg
    IndexVectorX GVGe;      ///< `|# of vertex-elems adjacencies|` element indices s.t.
                            ///< `GVGe[k] for GVGp[i] <= k < GVGp[i+1]` gives the element `e`
                            ///< adjacent to vertex `i`
    IndexVectorX GVGilocal; ///< `|# of vertex-elems adjacencies|` local vertex indices s.t.
                            ///< `GVGilocal[k] for GVGp[i] <= k < GVGp[i+1]` gives the local vertex
                            ///< index of vertex `i` in element `e=GVGe[k]`
    // Parallelization
    IndexVectorX colors; ///< `|# vertices|` map of vertex colors
    IndexVectorX Pptr;   ///< `|# partitions+1|` partition pointers, s.t. the range `[Pptr[p],
                         ///< Pptr[p+1])` indexes into Padj from partition `p`
    IndexVectorX Padj;   ///< `|# verts|` partition vertices
    // Time integration optimization parameters
    EInitializationStrategy strategy{
        EInitializationStrategy::Inertia}; ///< BCD optimization initialization strategy
    Scalar detHZero{1e-7};                 ///< Numerical zero for hessian pseudo-singularity check
};

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_CORE_H
