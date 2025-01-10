#include "HyperReduction.h"

#include "pbat/common/ArgSort.h"
#include "pbat/common/Indexing.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/graph/Partition.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/vbd/Data.h"
#include "pbat/sim/vbd/Mesh.h"

#include <ranges>
#include <tbb/parallel_for.h>
#include <utility>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

HyperReduction::HyperReduction(Data const& data, Index nTargetActiveElementsIn)
    : bActiveE(BoolVectorX::Constant(data.E.cols(), true)),
      bActiveK(BoolVectorX::Constant(data.X.cols(), true)),
      wgE(data.wg),
      mK(data.m),
      nTargetActiveElements(nTargetActiveElementsIn >= 0 ? nTargetActiveElementsIn : data.E.cols())
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.vbd.multigrid.Level.HyperReduction.Construct");
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat
