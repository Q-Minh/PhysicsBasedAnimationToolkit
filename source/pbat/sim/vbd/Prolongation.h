#ifndef PBAT_SIM_VBD_PROLONGATION_H
#define PBAT_SIM_VBD_PROLONGATION_H

#include "pbat/Aliases.h"

#include <cassert>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

struct Hierarchy;

struct Prolongation
{
    /**
     * @brief
     *
     * @param lc Coarse level
     * @return Prolongation&
     */
    Prolongation& From(Index lc);
    /**
     * @brief
     *
     * @param lf Fine level
     * @return Prolongation&
     */
    Prolongation& To(Index lf);
    /**
     * @brief
     *
     * @param ec |#verts at fine level| array of coarse cage elements containing fine level vertices
     * @param Nc 4x|#verts at fine level| array of coarse cage shape functions at fine level
     * vertices
     * @return Prolongation&
     */
    Prolongation& WithCoarseShapeFunctions(
        Eigen::Ref<IndexVectorX const> const& ec,
        Eigen::Ref<MatrixX const> const& Nc);
    /**
     * @brief
     *
     * @param bValidate Throw on ill-formed input
     * @return Prolongation&
     */
    Prolongation& Construct(bool bValidate = true);

    IndexVectorX
        ec; ///< |#verts at fine level| array of coarse cage elements containing fine level vertices
    MatrixX Nc;   ///< 4x|#verts at fine level| array of coarse cage shape functions at fine level
                  ///< vertices
    Index lc, lf; ///< Coarse (lc) and fine (lf) indices to levels in the hierarchy (lc > lf)

    /**
     * @brief Prolong level lc to level lf
     *
     * @param H
     */
    void Apply(Hierarchy& H);
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_PROLONGATION_H