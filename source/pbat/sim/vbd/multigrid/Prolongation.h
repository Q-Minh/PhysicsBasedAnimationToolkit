#ifndef PBAT_SIM_VBD_MULTIGRID_PROLONGATION_H
#define PBAT_SIM_VBD_MULTIGRID_PROLONGATION_H

#include "Quadrature.h"
#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Data;

namespace multigrid {

struct Level;

struct Prolongation
{
    Prolongation() = default;
    /**
     * @brief
     * @param FM
     * @param CM
     */
    Prolongation(VolumeMesh const& FM, VolumeMesh const& CM);
    /**
     * @brief Prolong coarse level lc to fine level lf
     * @param iters
     * @param lc
     * @param lf
     */
    void Apply(Level const& lc, Level& lf) const;
    /**
     * @brief
     * @param lc
     * @param lf
     */
    void Apply(Level const& lc, Data& lf) const;
    /**
     * @brief
     * @param lc
     * @param xf
     */
    void DoApply(Level const& lc, Eigen::Ref<MatrixX> xf) const;

    IndexVectorX ec; ///< |#fine verts| array of coarse cage elements containing fine mesh vertices
    MatrixX Nc;      ///< 4x|#fine verts| array of coarse cage shape functions at fine mesh vertices
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_PROLONGATION_H