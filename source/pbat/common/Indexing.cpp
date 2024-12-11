#include "Indexing.h"

#include <Eigen/Core>
#include <array>
#include <doctest/doctest.h>
#include <ranges>

namespace pbat {
namespace common {

TEST_CASE("[common] Cumulative sums are computable from any integral range type")
{
    std::array<Index, 3> v{5, 10, 15};
    auto const cs = CumSum(v);
    IndexVectorX csExpected(4);
    csExpected << 0, 5, 15, 30;
    CHECK_EQ(cs, csExpected);
}

} // namespace common
} // namespace pbat