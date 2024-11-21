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
    IndexVectorX
        ef; ///< |#verts at fine level| array of coarse cage elements containing fine level vertices
    MatrixX Nf;   ///< 4x|#verts at fine level| array of coarse cage shape functions at fine level
                  ///< vertices
    Index lc, lf; ///< Coarse (lc) and fine (lf) indices to levels in the hierarchy (lc > lf)

    void Apply(Hierarchy& H);
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_PROLONGATION_H