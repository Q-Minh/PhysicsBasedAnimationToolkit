#include "Cube.h"

namespace pbat {
namespace geometry {
namespace model {

std::pair<MatrixX, IndexMatrixX> Cube(EMesh mesh)
{
    MatrixX V{};
    IndexMatrixX C{};
    if (mesh == EMesh::Tetrahedral)
    {
        V.resize(3, 8);
        C.resize(4, 5);
        // clang-format off
        V << 0., 1., 0., 1., 0., 1., 0., 1.,
            0., 0., 1., 1., 0., 0., 1., 1.,
            0., 0., 0., 0., 1., 1., 1., 1.;
        C << 0, 3, 5, 6, 0,
            1, 2, 4, 7, 5,
            3, 0, 6, 5, 3,
            5, 6, 0, 3, 6;
        // clang-format on
    }
    return {V, C};
}

} // namespace model
} // namespace geometry
} // namespace pbat
