#ifndef PBAT_SIM_VBD_RESTRICTION_H
#define PBAT_SIM_VBD_RESTRICTION_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Hierarchy;

struct Restriction
{
    MatrixX Nfg; ///< 4x|#quad.pts.| array of fine cage shape functions at coarse quadrature points
    IndexVectorX
        efg; ///< |#quad.pts.| array of fine cage elements associated with quadrature points
    MatrixX xfg; ///< 3x|#quad.pts.| array of fine cage shape target positions at quadrature points
    Index iterations; ///< Number of BCD iterations to compute restriction
    Index lc, lf; ///< Indices to coarse (lc) and fine (lf) levels, respectively, assuming lc > lf

    void Apply(Hierarchy& H);
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_RESTRICTION_H