#ifndef PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H
#define PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

struct Data;

namespace multigrid {

struct HyperReduction
{
    using BoolVectorX = Eigen::Vector<bool, Eigen::Dynamic>;

    HyperReduction(
        Data const& data,
        Index nTargetActiveElements = Index(-1));

    BoolVectorX bActiveE; ///<
    BoolVectorX bActiveK; ///<

    VectorX wgE; ///<
    VectorX mK;  ///<

    Index nTargetActiveElements; ///< 
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_HYPERREDUCTION_H
