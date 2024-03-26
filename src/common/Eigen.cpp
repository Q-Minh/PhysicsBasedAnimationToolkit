#include "pba/common/Eigen.h"

#include <doctest/doctest.h>

namespace pba {
namespace common {

TEST_CASE("Conversions from STL to Eigen")
{
    std::vector<Scalar> v{1., 2., 3.};
    auto const veigen = ToEigen(v);
    CHECK_EQ(veigen.size(), v.size());
    for (auto i = 0u; i < veigen.size(); ++i)
        CHECK_EQ(v[i], veigen(i));
}

} // namespace common
} // namespace pba