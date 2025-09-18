#include "MultibodyTetrahedralMeshSystem.h"

namespace pbat::sim::contact {
} // namespace pbat::sim::contact

#include "pbat/geometry/model/Cube.h"

#include <algorithm>
#include <doctest/doctest.h>
#include <unordered_set>

TEST_CASE("[sim][contact] MultibodyTetrahedralMeshSystem")
{
    using namespace pbat::geometry::model;
    auto const [V1, T1] = Cube(EMesh::Tetrahedral, 0);
    auto const [V2, T2] = Cube(EMesh::Tetrahedral, 2);

    auto const fAssert = [](pbat::IndexMatrixX const& T,
                            pbat::sim::contact::MultibodyTetrahedralMeshSystem<> const& system) {
        bool const bNodesAreSortedByConnectedComponent =
            std::is_sorted(system.CC.data(), system.CC.data() + system.CC.size());
        CHECK(bNodesAreSortedByConnectedComponent);
        bool const bEdgesAreSortedByConnectedComponent =
            std::is_sorted(system.E.data(), system.E.data() + system.E.size(), [&](auto i, auto j) {
                return system.CC[i] < system.CC[j];
            });
        CHECK(bEdgesAreSortedByConnectedComponent);
        bool const bTrianglesAreSortedByConnectedComponent =
            std::is_sorted(system.F.data(), system.F.data() + system.F.size(), [&](auto i, auto j) {
                return system.CC[i] < system.CC[j];
            });
        CHECK(bTrianglesAreSortedByConnectedComponent);
        bool const bTetsAreSortedByConnectedComponent =
            std::is_sorted(T.data(), T.data() + T.size(), [&](auto i, auto j) {
                return system.CC[i] < system.CC[j];
            });
        CHECK(bTetsAreSortedByConnectedComponent);
        CHECK_EQ(system.NumBodies(), 2);
        CHECK_EQ(system.VP.tail<1>()(0), system.V.size());
        CHECK_EQ(system.EP.tail<1>()(0), system.E.cols());
        CHECK_EQ(system.FP.tail<1>()(0), system.F.cols());
        CHECK_EQ(system.TP.tail<1>()(0), T.cols());

        bool const bVertexPrefixSumIsUnique =
            std::unordered_set<pbat::Index>(system.VP.data(), system.VP.data() + system.VP.size())
                .size() == static_cast<std::size_t>(system.VP.size());
        CHECK(bVertexPrefixSumIsUnique);
        bool const bEdgePrefixSumIsUnique =
            std::unordered_set<pbat::Index>(system.EP.data(), system.EP.data() + system.EP.size())
                .size() == static_cast<std::size_t>(system.EP.size());
        CHECK(bEdgePrefixSumIsUnique);
        bool const bTrianglePrefixSumIsUnique =
            std::unordered_set<pbat::Index>(system.FP.data(), system.FP.data() + system.FP.size())
                .size() == static_cast<std::size_t>(system.FP.size());
        CHECK(bTrianglePrefixSumIsUnique);
        bool const bTetrahedronPrefixSumIsUnique =
            std::unordered_set<pbat::Index>(system.TP.data(), system.TP.data() + system.TP.size())
                .size() == static_cast<std::size_t>(system.TP.size());
        CHECK(bTetrahedronPrefixSumIsUnique);
    };

    SUBCASE("Already sorted system")
    {
        // Arrange
        pbat::MatrixX X(3, V1.cols() + V2.cols());
        X << V1, V2;
        pbat::IndexMatrixX T(4, T1.cols() + T2.cols());
        T << T1, T2.array() + V1.cols();
        // Act
        pbat::sim::contact::MultibodyTetrahedralMeshSystem<> system{X, T};
        // Assert
        fAssert(T, system);
    }
    SUBCASE("Unsorted system")
    {
        // Arrange
        pbat::MatrixX X(3, V1.cols() + V2.cols());
        X << V1, V2;
        pbat::IndexMatrixX T(4, T1.cols() + T2.cols());
        T << T1, T2.array() + V1.cols();
        T.col(0).swap(T.col(T.cols() - 1)); // Swap the first tetrahedron with the last one
        // Act
        pbat::sim::contact::MultibodyTetrahedralMeshSystem<> system{X, T};
        // Assert
        fAssert(T, system);
    }
}