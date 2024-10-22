#ifndef PBAT_GPU_VBD_TESTS_FEM_H
#define PBAT_GPU_VBD_TESTS_FEM_H

#include "pbat/gpu/Aliases.h"

namespace pbat {
namespace gpu {
namespace vbd {
namespace tests {

class LinearFemMesh
{
  public:
    LinearFemMesh(
        Eigen::Ref<GpuMatrixX const> const& V,
        Eigen::Ref<GpuIndexMatrixX const> const& T);
    LinearFemMesh(LinearFemMesh const&)            = delete;
    LinearFemMesh& operator=(LinearFemMesh const&) = delete;
    ~LinearFemMesh();

    GpuVectorX QuadratureWeights() const;
    GpuMatrixX ShapeFunctionGradients() const;
    GpuMatrixX LameCoefficients(GpuScalar Y, GpuScalar nu) const;

  private:
    void* mImpl;
};

} // namespace tests
} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_TESTS_FEM_H