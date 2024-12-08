#ifndef PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H
#define PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H

#include "Level.h"
#include "Mesh.h"
#include "Quadrature.h"
#include "pbat/sim/vbd/Data.h"

#include <optional>
#include <utility>
#include <vector>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

struct Hierarchy
{
    Hierarchy(
        Data root,
        std::vector<Eigen::Ref<MatrixX const>> const& X,
        std::vector<Eigen::Ref<IndexMatrixX const>> const& E,
        std::optional<std::vector<std::pair<Index, Index>>> cycle = std::nullopt,
        std::optional<std::vector<Index>> transitionSchedule      = std::nullopt,
        std::optional<std::vector<Index>> smoothingSchedule       = std::nullopt);

    Data root;
    std::vector<Level> levels;
    std::vector<std::pair<Index, Index>> cycle;
    std::vector<Index> smoothingSchedule;
    std::vector<Index> transitionSchedule;
};

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_MULTIGRID_HIERARCHY_H