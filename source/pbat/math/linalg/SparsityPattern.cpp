#include "SparsityPattern.h"

#include <doctest/doctest.h>
#include <ranges>
#include <vector>
#include <iostream>

TEST_CASE("[math][linalg] SparsityPattern")
{
    using namespace pbat;

    auto constexpr nRows = 4;
    auto constexpr nCols = 5;
    MatrixX Aexpected(nRows, nCols);
    // clang-format off
    Aexpected << 0, 0, 3, 0, 0,
                 2, 0, 1, 0, 1,
                 0, 1, 0, 0, 2,
                 0, 0, 0, 0, 0;
    // clang-format on
    std::vector<Index> rowIndices{{1, 1, 2, 0, 0, 0, 1, 1, 2, 2}};
    std::vector<Index> colIndices{{0, 0, 1, 2, 2, 2, 2, 4, 4, 4}};
    std::vector<Scalar> nonZeros(10, 1.);

    math::linalg::SparsityPattern const sparsityPattern(nRows, nCols, rowIndices, colIndices);
    bool const bIsEmpty = sparsityPattern.IsEmpty();
    CHECK_FALSE(bIsEmpty);
    CSCMatrix const A = sparsityPattern.ToMatrix(nonZeros);
    CHECK_EQ(A.rows(), nRows);
    CHECK_EQ(A.cols(), nCols);
    IndexVector<5> columnCounts{};
    columnCounts << 1, 1, 2, 0, 2;
    for (auto j = 0; j < nCols; ++j)
    {
        CHECK_EQ(A.col(j).nonZeros(), columnCounts(j));
    }

    MatrixX const Adense  = A.toDense();
    Scalar const error    = (Adense - Aexpected).norm() / Aexpected.norm();
    Scalar constexpr zero = 1e-15;
    CHECK_LE(error, zero);
}
