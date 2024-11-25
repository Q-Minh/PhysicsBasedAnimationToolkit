#include "Hierarchy.h"

namespace pbat {
namespace sim {
namespace vbd {

Hierarchy::Hierarchy(
    Data rootIn,
    std::vector<Level> levelsIn,
    std::vector<Smoother> smoothersIn,
    std::vector<MatrixX> NlIn,
    std::vector<Transition> transitionsIn)
    : root(std::move(rootIn)),
      levels(std::move(levelsIn)),
      smoothers(std::move(smoothersIn)),
      Nl(std::move(NlIn)),
      transitions(std::move(transitionsIn))
{
}

} // namespace vbd
} // namespace sim
} // namespace pbat