#ifndef PBAT_GPU_GEOMETRY_PRIMITIVES_CUH
#define PBAT_GPU_GEOMETRY_PRIMITIVES_CUH

#define EIGEN_NO_CUDA
#include "pbat/Aliases.h"
#undef EIGEN_NO_CUDA

#include "pbat/gpu/Aliases.h"

#include <thrust/device_vector.h>

namespace pbat {
namespace gpu {
namespace geometry {

struct Points
{
    Points(Eigen::Ref<MatrixX const> const& V);

    thrust::device_vector<GpuScalar> x, y, z; ///< Point coordinates
};

struct Simplices
{
    /**
     * @brief Type of mesh simplex. The enum's integer value reveals the number of vertices which
     * form the simplex.
     */
    enum class ESimplexType : int { Edge = 2, Triangle = 3, Tetrahedron = 4 };

    Simplices(Eigen::Ref<IndexMatrixX const> const& C);

    GpuIndex NumberOfSimplices() const
    {
        return static_cast<GpuIndex>(inds.size()) / static_cast<GpuIndex>(eSimplexType);
    }

    ESimplexType eSimplexType;            ///< Type of simplex stored in inds
    thrust::device_vector<GpuIndex> inds; ///< Flattened array of simplex indices, where the range
                                          ///< [ inds[i*eSimplexType], inds[(i+1)*eSimplexType] )
                                          ///< yields the vertices of the i^{th} simplex
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_PRIMITIVES_CUH