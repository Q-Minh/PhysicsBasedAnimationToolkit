#include "MultibodyTetrahedralMeshSystem.h"

namespace pbat::sim::contact {
} // namespace pbat::sim::contact

#include "pbat/geometry/model/Cube.h"

#include <doctest/doctest.h>

TEST_CASE("[sim][contact] MultibodyTetrahedralMeshSystem")
{
    using namespace pbat::geometry::model;
    auto const [V1, T1] = Cube(EMesh::Tetrahedral, 0);
    auto const [V2, T2] = Cube(EMesh::Tetrahedral, 2);

    SUBCASE("Already sorted system")
    {
        // Arrange
        pbat::MatrixX X(3, V1.cols() + V2.cols());
        X << V1, V2;
        pbat::IndexMatrixX T(4, T1.cols() + T2.cols());
        T << T1, T2.array() + V1.cols();
        // Act
        pbat::sim::contact::MultibodyTetrahedralMeshSystem<> system{};
        system.Construct<pbat::Scalar>(X, T);
        // Assert
        CHECK_EQ(system.NumberOfBodies(), 2);
        CHECK_EQ(system.VP.tail<1>()(0), system.V.size());
        CHECK_EQ(system.EP.tail<1>()(0), system.E.cols());
        CHECK_EQ(system.FP.tail<1>()(0), system.F.cols());
        CHECK_EQ(system.TP.tail<1>()(0), T.cols());
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
        pbat::sim::contact::MultibodyTetrahedralMeshSystem<> system{};
        system.Construct<pbat::Scalar>(X, T);
        // Assert
        CHECK_EQ(system.NumberOfBodies(), 2);
        CHECK_EQ(system.VP.tail<1>()(0), system.V.size());
        CHECK_EQ(system.EP.tail<1>()(0), system.E.cols());
        CHECK_EQ(system.FP.tail<1>()(0), system.F.cols());
        CHECK_EQ(system.TP.tail<1>()(0), T.cols());
    }
}