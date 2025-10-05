#include "Permute.h"

#include "ArgSort.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[common] Permute")
{
    using namespace pbat;
    // Arrange
    Index constexpr n = 10;
    VectorX values    = VectorX::Random(n);
    IndexVectorX permutation =
        common::ArgSort(n, [&](Index i, Index j) { return values(i) < values(j); });
    IndexVectorX permutationExpected = permutation;
    // Act
    common::Permute(values.begin(), values.end(), permutation.begin());
    // Assert
    CHECK(permutation == permutationExpected);
    bool const bIsSorted = std::is_sorted(values.begin(), values.end());
    CHECK(bIsSorted);
}