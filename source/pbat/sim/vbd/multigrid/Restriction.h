#ifndef PBAT_SIM_VBD_MULTIGRID_RESTRICTION_H
#define PBAT_SIM_VBD_MULTIGRID_RESTRICTION_H

#include "Quadrature.h"
#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Level;

struct Restriction
{
    Restriction() = default;
    /**
     * @brief
     * @param CQ
     */
    Restriction(CageQuadrature const& CQ);
    /**
     * @brief Restrict fine level lf to coarse level lc
     * @param iters
     * @param lf
     * @param lc
     */
    void Apply(Index iters, Level const& lf, Level& lc);
    /**
     * @brief
     * @param iters
     * @param xf
     * @param lc
     * @return
     */
    Scalar DoApply(
        Index iters,
        Eigen::Ref<MatrixX const> const& xf,
        Eigen::Ref<IndexMatrixX const> const& Ef,
        Level& lc);

    MatrixX xfg; ///< 3x|#quad.pts.| array of fine cage shape target positions at quad.pts.
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_RESTRICTION_H