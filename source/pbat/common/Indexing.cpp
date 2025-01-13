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

TEST_CASE("[common] Repeat on Eigen vectors works as expected")
{
    using namespace pbat;
    VectorX x(3);
    x << 1.0, 2.0, 3.0;

    IndexVectorX r(3);
    r << 2, 3, 1;

    VectorX expected(6);
    expected << 1.0, 1.0, 2.0, 2.0, 2.0, 3.0;

    VectorX result = common::Repeat(x, r);

    CHECK_EQ(result.size(), expected.size());
    bool const bAreEqual = (result.array() == expected.array()).all();
    CHECK(bAreEqual);
}

} // namespace common
} // namespace pbat