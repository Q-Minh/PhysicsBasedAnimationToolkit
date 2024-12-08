#include "Hierarchy.h"

#include "Mesh.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"

#include <exception>
#include <fmt/format.h>
#include <string>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

Hierarchy::Hierarchy(
    Data rootIn,
    std::vector<Eigen::Ref<MatrixX const>> const& X,
    std::vector<Eigen::Ref<IndexMatrixX const>> const& E,
    std::optional<std::vector<std::pair<Index, Index>>> cycleIn,
    std::optional<std::vector<Index>> transitionScheduleIn,
    std::optional<std::vector<Index>> smoothingScheduleIn)
    : root(std::move(rootIn)), levels(), cycle(), smoothingSchedule(), transitionSchedule()
{
    levels.reserve(X.size());
    for (auto l = 0ULL; l < X.size(); ++l)
    {
        levels.push_back(
            Level(VolumeMesh(X[l], E[l]))
                .WithCageQuadrature(root, ECageQuadratureStrategy::PolynomialSubCellIntegration)
                .WithDirichletQuadrature(root)
                .WithMomentumEnergy(root)
                .WithElasticEnergy(root));
    }
    if (cycleIn)
    {
        cycle = std::move(*cycleIn);
    }
    else
    {
        cycle.reserve(levels.size() + 1);
        Index const nLevels = static_cast<Index>(levels.size());
        cycle.push_back({Index(-1), nLevels - 1});
        for (Index l = nLevels - 1; l >= 0; --l)
            cycle.push_back({l, l - 1});
    }
    if (smoothingScheduleIn)
    {
        smoothingSchedule = std::move(*smoothingScheduleIn);
    }
    else
    {
        smoothingSchedule.resize(cycle.size() + 1, Index(10));
    }
    if (transitionScheduleIn)
    {
        transitionSchedule = std::move(*transitionScheduleIn);
    }
    else
    {
        transitionSchedule.resize(cycle.size(), Index(10));
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat