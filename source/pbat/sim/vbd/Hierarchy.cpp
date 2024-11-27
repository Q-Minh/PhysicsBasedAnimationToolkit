#include "Hierarchy.h"

namespace pbat {
namespace sim {
namespace vbd {

Hierarchy::Hierarchy(
    Data rootIn,
    std::vector<Level> levelsIn,
    std::vector<Transition> transitionsIn,
    std::vector<Smoother> smoothersIn)
    : root(std::move(rootIn)),
      levels(std::move(levelsIn)),
      transitions(std::move(transitionsIn)),
      smoothers(std::move(smoothersIn))
{
}

} // namespace vbd
} // namespace sim
} // namespace pbat