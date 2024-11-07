#ifndef PBAT_GPU_VBD_TESTS_FEM_H
#define PBAT_GPU_VBD_TESTS_FEM_H

#include "pbat/Aliases.h"

namespace pbat {
namespace gpu {
namespace vbd {
namespace tests {

class LinearFemMesh
{
  public:
    LinearFemMesh(Eigen::Ref<MatrixX const> const& V, Eigen::Ref<IndexMatrixX const> const& T);
    LinearFemMesh(LinearFemMesh const&)            = delete;
    LinearFemMesh& operator=(LinearFemMesh const&) = delete;
    ~LinearFemMesh();

    VectorX QuadratureWeights() const;
    MatrixX ShapeFunctionGradients() const;
    MatrixX LameCoefficients(Scalar Y, Scalar nu) const;

  private:
    void* mImpl;
};

} // namespace tests
} // namespace vbd
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_VBD_TESTS_FEM_H