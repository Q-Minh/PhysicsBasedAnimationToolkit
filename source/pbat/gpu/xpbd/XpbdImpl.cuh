#ifndef PBAT_GPU_XPBD_XPBD_IMPL_CUH
#define PBAT_GPU_XPBD_XPBD_IMPL_CUH

#define EIGEN_NO_CUDA
#include "pbat/Aliases.h"
#undef EIGEN_NO_CUDA

#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/Buffer.cuh"
#include "pbat/gpu/common/Var.cuh"
#include "pbat/gpu/geometry/PrimitivesImpl.cuh"

#include <cstddef>
#include <vector>

namespace pbat {
namespace gpu {
namespace xpbd {

class XpbdImpl
{
  public:
    /**
     * @brief
     * @param dt
     */
    void Step(GpuScalar dt);
    /**
     * @brief
     * @return
     */
    std::size_t NumberOfParticles() const;
    /**
     * @brief
     * @return
     */
    std::size_t NumberOfConstraints() const;

  private:
    std::vector<common::Buffer<GpuIndex>> mPartitions; ///< Constraint partitions
    geometry::PointsImpl mV;                           ///< Vertex/particle positions
    geometry::SimplicesImpl mT;                        ///< Tetrahedral simplices

    common::Buffer<GpuScalar, 3> xt;    ///< Vertex/particle positions at time t
    common::Buffer<GpuScalar, 3> v;     ///< Vertex/particle velocities
    common::Buffer<GpuScalar, 3> f;     ///< Vertex/particle external forces
    common::Buffer<GpuScalar> m;        ///< Vertex/particle masses
    common::Buffer<GpuScalar, 2> lame;  ///< Lame coefficients
    common::Buffer<GpuScalar> DmInv;    ///< 3x3x|#elements| array of material shape matrix inverses
    common::Buffer<GpuScalar> lambda;   ///< "Lagrange" multipliers
    common::Buffer<GpuScalar, 2> alpha; ///< Compliance

    GpuIndex S;    ///< Number of substeps per timestep
    GpuIndex K;    ///< Maximum number of iterations per constraint solve
    GpuScalar muf; ///< Coulomb friction coefficient
    GpuScalar muc; ///< Collision penalty
};

} // namespace xpbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_XPBD_XPBD_IMPL_CUH