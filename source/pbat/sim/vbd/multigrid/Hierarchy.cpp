#include "Hierarchy.h"

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

Hierarchy::Hierarchy(
    Data dataIn,
    std::vector<VolumeMesh> cages,
    IndexVectorX const& cycleIn,
    IndexVectorX const& sitersIn)
    : data(std::move(dataIn)), levels(), cycle(cycleIn), siters(sitersIn)
{
    levels.reserve(cages.size());
    for (VolumeMesh& cage : cages)
        levels.push_back(Level(data, std::move(cage)));
    // Reasonable defaults
    if (cycle.size() == 0)
    {
        // Standard v-cycle
        Index nLevels = static_cast<Index>(levels.size());
        cycle.resize(nLevels * 2 + 1);
        Index k    = 0;
        cycle(k++) = Index(-1);
        for (Index l = 0; l < nLevels; ++l)
            cycle(k++) = l;
        for (Index l = 0; l < nLevels; ++l)
            cycle(k++) = nLevels - l - 2;
    }
    if (siters.size() == 0)
    {
        // TODO: Find better strategy for default iterations!
        siters.setConstant(static_cast<Index>(cycle.size()), 5);
    }
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][multigrid] Hierarchy")
{
    using namespace pbat;
    using sim::vbd::Data;
    using sim::vbd::VolumeMesh;
    using sim::vbd::multigrid::Hierarchy;
    // Arrange
    auto const [VR, CR]   = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 2);
    Data data             = Data().WithVolumeMesh(VR, CR).Construct();
    auto const [VL2, CL2] = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 0);
    auto const [VL1, CL1] = geometry::model::Cube(geometry::model::EMesh::Tetrahedral, 1);

    // Act
    Hierarchy H{std::move(data), {VolumeMesh(VL1, CL1), VolumeMesh(VL2, CL2)}};

    // Assert
    CHECK_EQ(H.siters.size(), H.cycle.size());
}
