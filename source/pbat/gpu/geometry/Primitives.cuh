#ifndef PBAT_GPU_GEOMETRY_PRIMITIVES_CUH
#define PBAT_GPU_GEOMETRY_PRIMITIVES_CUH

#define EIGEN_NO_CUDA
#include "pbat/Aliases.h"
#undef EIGEN_NO_CUDA

#include <thrust/device_vector.h>

namespace pbat {
namespace gpu {
namespace geometry {

struct Points
{
    using ScalarType = float;

    Points(Eigen::Ref<MatrixX const> const& V);

    thrust::device_vector<ScalarType> x, y, z; ///< Point coordinates
};

struct Simplices
{
    using IndexType = int;

    enum ESimplexType : int {
        Vertex      = 1 << 0,
        Edge        = 1 << 1,
        Triangle    = 1 << 2,
        Tetrahedron = 1 << 3
    };

    Simplices(Eigen::Ref<IndexMatrixX const> const& C, int simplexTypes);

    int eSimplexTypes; ///< Bitmask revealing which simplex cell types are stored in (c,inds)
    thrust::device_vector<IndexType>
        c; ///< |# cells + 1| array of cell index beginnings into inds (prefix sum), where a cell in
           ///< cells may be any of ESimplexType
    thrust::device_vector<IndexType>
        inds; ///< Flattened array of cell indices, where the range [ inds[c[i]], inds[c[i+1]] )
              ///< yields the vertices of the i^{th} cell
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_PRIMITIVES_CUH