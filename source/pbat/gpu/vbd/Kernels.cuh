#ifndef PBAT_GPU_VBD_KERNELS_CUH
#define PBAT_GPU_VBD_KERNELS_CUH

#include "pbat/HostDevice.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <array>
#include <cstddef>
#include <limits>

namespace pbat {
namespace gpu {
namespace vbd {
namespace kernels {

struct BackwardEulerMinimization
{
    GpuScalar dt;                     ///< Time step
    GpuScalar dt2;                    ///< Squared time step
    GpuScalar* m;                     ///< Lumped mass matrix
    std::array<GpuScalar*, 3> xtilde; ///< Inertial target
    std::array<GpuScalar*, 3> xt;     ///< Previous vertex positions
    std::array<GpuScalar*, 3> x;      ///< Vertex positions

    std::array<GpuIndex*, 4> T; ///< 4x|#elements| array of tetrahedra
    GpuScalar* wg;              ///< |#elements| array of quadrature weights
    GpuScalar* GP;              ///< 4x3x|#elements| array of shape function gradients
    GpuScalar* lame;            ///< 2x|#elements| of 1st and 2nd Lame coefficients
    GpuScalar detHZero;         ///< Numerical zero for hessian determinant check
    // GpuScalar const* kD;                  ///< |#elements| array of damping coefficients

    GpuIndex* GVTp;      ///< Vertex-tetrahedron adjacency list's prefix sum
    GpuIndex* GVTn;      ///< Vertex-tetrahedron adjacency list's neighbour list
    GpuIndex* GVTilocal; ///< Vertex-tetrahedron adjacency list's ilocal property

    GpuScalar kD;                             ///< Rayleigh damping coefficient
    GpuScalar kC;                             ///< Collision penalty
    GpuIndex nMaxCollidingTrianglesPerVertex; ///< Memory capacity for storing vertex triangle
                                              ///< collision constraints
    GpuIndex* FC; ///< |#vertices|x|nMaxCollidingTrianglesPerVertex| array of colliding triangles
    GpuIndex* nCollidingTriangles; ///< |#vertices| array of the number of colliding triangles
                                   ///< for each vertex.
    std::array<GpuIndex*, 4> F;    ///< 3x|#collision triangles| array of triangles

    GpuIndex*
        partition; ///< List of vertex indices that can be processed independently, i.e. in parallel

    auto constexpr ExpectedSharedMemoryPerThreadInBytes() const { return 12 * sizeof(GpuScalar); }
};

__global__ void MinimizeBackwardEuler(BackwardEulerMinimization BDF);

} // namespace kernels
} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_KERNELS_CUH