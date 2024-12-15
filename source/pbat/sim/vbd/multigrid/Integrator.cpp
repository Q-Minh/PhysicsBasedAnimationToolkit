#include "Integrator.h"

#include "Hierarchy.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

void Integrator::Step(Scalar dt, Index substeps, Hierarchy& hierarchy) const
{
    Scalar sdt = dt / static_cast<Scalar>(substeps);
    for (Index s = 0; s < substeps; ++s)
    {
        auto nLevelVisits = hierarchy.cycle.size();
        for (auto c = 0; c < nLevelVisits; ++c)
        {
            auto l      = static_cast<std::size_t>(hierarchy.cycle(c));
            Index iters = hierarchy.siters(c);
            hierarchy.levels[l].Smooth(sdt, iters, hierarchy.data);
        }
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat
