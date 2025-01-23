#ifndef PBAT_GPU_IMPL_GEOMETRY_PRIMITIVES_CUH
#define PBAT_GPU_IMPL_GEOMETRY_PRIMITIVES_CUH

#include "pbat/Aliases.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/impl/common/Buffer.cuh"

#include <array>

namespace pbat {
namespace gpu {
namespace impl {
namespace geometry {

struct Points
{
    /**
     * @brief
     * @param V |#dims|x|#vertices| array of vertex positions
     */
    Points(Eigen::Ref<GpuMatrixX const> const& V);
    /**
     * @brief
     * @return
     */
    std::size_t NumberOfPoints() const;
    /**
     * @brief
     * @return
     */
    std::size_t Dimensions() const;
    /**
     * @brief
     * @param V |#dims|x|#vertices| array of vertex positions
     */
    void Update(Eigen::Ref<GpuMatrixX const> const& V);

    common::Buffer<GpuScalar, 3> x; ///< Point coordinates
};

struct Simplices
{
    /**
     * @brief Type of mesh simplex. The enum's integer value reveals the number of vertices which
     * form the simplex.
     */
    enum class ESimplexType : int { Vertex = 1, Edge = 2, Triangle = 3, Tetrahedron = 4 };

    Simplices(Eigen::Ref<GpuIndexMatrixX const> const& C);
    /**
     * @brief
     * @return
     */
    GpuIndex NumberOfSimplices() const;

    ESimplexType eSimplexType; ///< Type of simplex stored in inds
    common::Buffer<GpuIndex, 4>
        inds; ///< Array of simplex vertex indices. inds[m][i] yields the index of the m^{th} vertex
              ///< of the i^{th} simplex. If m is >= than the simplex's dimensionality, i.e. m >=
              ///< eSimplexType, then inds[m][i] = -inds[0][i] - 1. This property ensures that for
              ///< any 2 simplices i,j of any dimensionality, if i,j are not topologically adjacent,
              ///< then inds[m][i] != inds[m][j] for m=0,1,2,3.
};

class Bodies
{
  public:
    Bodies(Eigen::Ref<GpuIndexVectorX const> const& B);

    GpuIndex NumberOfBodies() const;

    common::Buffer<GpuIndex>
        body; ///< |#simplices|x1 array of body indices, such that body[s] -> body of simplex s.

  private:
    GpuIndex nBodies;
};

} // namespace geometry
} // namespace impl
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_IMPL_GEOMETRY_PRIMITIVES_CUH