#ifndef PBAT_GPU_GEOMETRY_PRIMITIVES_IMPL_CUH
#define PBAT_GPU_GEOMETRY_PRIMITIVES_IMPL_CUH

#define EIGEN_NO_CUDA
#include "pbat/Aliases.h"
#undef EIGEN_NO_CUDA

#include "pbat/gpu/Aliases.h"

#include <array>
#include <thrust/device_vector.h>

namespace pbat {
namespace gpu {
namespace geometry {

struct PointsImpl
{
    /**
     * @brief
     * @param V |#dims|x|#vertices| array of vertex positions
     */
    PointsImpl(Eigen::Ref<GpuMatrixX const> const& V);
    /**
     * @brief
     * @param V |#dims|x|#vertices| array of vertex positions
     */
    void Update(Eigen::Ref<GpuMatrixX const> const& V);
    /**
     * @brief Obtains the raw device pointers to the point coordinates
     * @return
     */
    std::array<GpuScalar const*, 3> Raw() const;
    /**
     * @brief Obtains the raw device pointers to the point coordinates
     * @return
     */
    std::array<GpuScalar*, 3> Raw();

    std::array<thrust::device_vector<GpuScalar>, 3> x; ///< Point coordinates
};

struct SimplicesImpl
{
    /**
     * @brief Type of mesh simplex. The enum's integer value reveals the number of vertices which
     * form the simplex.
     */
    enum class ESimplexType : int { Vertex = 1, Edge = 2, Triangle = 3, Tetrahedron = 4 };

    SimplicesImpl(Eigen::Ref<GpuIndexMatrixX const> const& C);
    /**
     * @brief
     * @return
     */
    GpuIndex NumberOfSimplices() const;
    /**
     * @brief Obtains the raw device pointers to the simplex indices
     * @return
     */
    std::array<GpuIndex const*, 4> Raw() const;
    /**
     * @brief Obtains the raw device pointers to the simplex indices
     * @return
     */
    std::array<GpuIndex*, 4> Raw();

    ESimplexType eSimplexType; ///< Type of simplex stored in inds
    std::array<thrust::device_vector<GpuIndex>, 4>
        inds; ///< Array of simplex vertex indices. inds[m][i] yields the index of the m^{th} vertex
              ///< of the i^{th} simplex. If m is >= than the simplex's dimensionality, i.e. m >=
              ///< eSimplexType, then inds[m][i] = -inds[0][i] - 1. This property ensures that for
              ///< any 2 simplices i,j of any dimensionality, if i,j are not topologically adjacent,
              ///< then inds[m][i] != inds[m][j] for m=0,1,2,3.
};

} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_PRIMITIVES_IMPL_CUH