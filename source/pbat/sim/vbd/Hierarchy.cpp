#include "Hierarchy.h"

namespace pbat {
namespace sim {
namespace vbd {

Hierarchy::Hierarchy(
    Data rootIn,
    std::vector<Level> levelsIn,
    std::vector<Smoother> smoothersIn,
    std::vector<MatrixX> NgIn,
    std::vector<Transition> transitionsIn)
    : root(std::move(rootIn)),
      levels(std::move(levelsIn)),
      smoothers(std::move(smoothersIn)),
      Ng(std::move(NgIn)),
      transitions(std::move(transitionsIn))
{
}

} // namespace vbd
} // namespace sim
} // namespace pbat