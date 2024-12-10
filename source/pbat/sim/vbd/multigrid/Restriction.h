#ifndef PBAT_SIM_VBD_MULTIGRID_RESTRICTION_H
#define PBAT_SIM_VBD_MULTIGRID_RESTRICTION_H

#include "Mesh.h"
#include "Quadrature.h"
#include "pbat/Aliases.h"
#include "pbat/sim/vbd/Data.h"

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
     * @param problem
     * @param FM
     * @param CM
     * @param CQ
     */
    Restriction(
        Data const& problem,
        VolumeMesh const& FM,
        VolumeMesh const& CM,
        CageQuadrature const& CQ);
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

    IndexVectorX efg; ///< |#quad.pts.| array of fine cage elements associated with quad.pts.
    MatrixX Nfg;      ///< 4x|#quad.pts.| array of fine cage shape functions at quad.pts.
    MatrixX xfg;      ///< 3x|#quad.pts.| array of fine cage shape target positions at quad.pts.
    VectorX rhog;     ///< |#quad.pts.| array of mass densities at quad.pts.
    VectorX mug;      ///< |#quad.pts.| array of 1st Lame coefficients at quad.pts.
    VectorX lambdag;  ///< |#quad.pts.| array of 2nd Lame coefficients at quad.pts.
    MatrixX Ncg;      ///< 4x|#quad.pts.| array of coarse cage shape functions at quad.pts.
    MatrixX
        GNcg; ///< 4x|3*#quad.pts.| array of coarse element shape function gradients at quad.pts.
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_RESTRICTION_H