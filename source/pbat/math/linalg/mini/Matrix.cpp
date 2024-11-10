#include "Matrix.h"

#include "BinaryOperations.h"
#include "Reductions.h"
#include "Repeat.h"
#include "Stack.h"
#include "pbat/Aliases.h"

#include <doctest/doctest.h>

TEST_CASE("[math][linalg][mini] Matrix")
{
    using namespace pbat::math::linalg::mini;
    auto constexpr kRows = 3;
    auto constexpr kCols = 3;
    using ScalarType     = pbat::Scalar;
    using MatrixType     = SMatrix<ScalarType, kRows, kCols>;

    PBAT_MINI_CHECK_READABLE_CONCEPTS(MatrixType);
    PBAT_MINI_CHECK_WRITEABLE_CONCEPTS(MatrixType);

    MatrixType M{};
    CHECK_EQ(M.Rows(), kRows);
    CHECK_EQ(M.Cols(), kCols);
    CHECK_EQ(M.Slice<2, 2>(1, 1)(0, 0), M(1, 1));
    CHECK_EQ(M.Col(2)(1), M(1, 2));
    CHECK_EQ(M.Row(2)(1), M(2, 1));

    using MatrixViewType = SMatrixView<ScalarType, kRows, kCols>;
    PBAT_MINI_CHECK_READABLE_CONCEPTS(MatrixViewType);
    PBAT_MINI_CHECK_WRITEABLE_CONCEPTS(MatrixViewType);

    pbat::MatrixX data(2, 10);
    data.leftCols(5).array()  = ScalarType(1);
    data.rightCols(5).array() = ScalarType(2);
    auto flatBufViewLeft      = FromFlatBuffer<2, 5>(data.data(), 0);
    auto flatBufViewRight     = FromFlatBuffer<2, 5>(data.data(), 1);
    CHECK(All(flatBufViewLeft == ScalarType(1)));
    CHECK(All(flatBufViewRight == ScalarType(2)));

    pbat::VectorX dataTop    = data.row(0);
    pbat::VectorX dataBottom = data.row(1);
    std::array<ScalarType*, 2> bufs{dataTop.data(), dataBottom.data()};
    auto bufViewLeft  = FromBuffers<2, 5>(bufs, 0);
    auto bufViewRight = FromBuffers<2, 5>(bufs, 1);
    CHECK(All(flatBufViewLeft == bufViewLeft));
    CHECK(All(flatBufViewRight == bufViewRight));
    ToBuffers(bufViewLeft + Ones<ScalarType, 2, 5>(), bufs, 0);
    ToBuffers(bufViewRight + Ones<ScalarType, 2, 5>(), bufs, 1);
    for (auto i = 0; i < 5; ++i)
    {
        CHECK_EQ(dataTop(i), ScalarType(2));
        CHECK_EQ(dataBottom(i), ScalarType(2));
    }
    for (auto i = 5; i < 10; ++i)
    {
        CHECK_EQ(dataTop(i), ScalarType(3));
        CHECK_EQ(dataBottom(i), ScalarType(3));
    }

    ToFlatBuffer(flatBufViewLeft + Ones<ScalarType, 2, 5>(), data.data(), 0);
    ToFlatBuffer(flatBufViewRight + Ones<ScalarType, 2, 5>(), data.data(), 1);
    for (auto i = 0; i < 5; ++i)
    {
        CHECK_EQ(data(0, i), ScalarType(2));
        CHECK_EQ(data(1, i), ScalarType(2));
    }
    for (auto i = 5; i < 10; ++i)
    {
        CHECK_EQ(data(0, i), ScalarType(3));
        CHECK_EQ(data(1, i), ScalarType(3));
    }

    using IndexType = int;
    SMatrix<IndexType, 1, 5> inds{0, 2, 4, 6, 8};
    ToFlatBuffer(Ones<ScalarType, 2, 5>(), inds, data.data());
    for (auto i = 0; i < 10; i += 2)
    {
        CHECK_EQ(data(0, i), ScalarType(1));
        CHECK_EQ(data(1, i), ScalarType(1));
    }

    ToBuffers(Ones<ScalarType, 2, 5>(), inds, bufs);
    for (auto i = 0; i < 10; i += 2)
    {
        CHECK_EQ(dataTop(i), ScalarType(1));
        CHECK_EQ(dataBottom(i), ScalarType(1));
    }
}