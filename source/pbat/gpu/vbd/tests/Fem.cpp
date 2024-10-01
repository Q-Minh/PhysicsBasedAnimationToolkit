#include "Fem.h"

#include "pbat/Aliases.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/Mesh.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/fem/Tetrahedron.h"
#include "pbat/physics/HyperElasticity.h"

namespace pbat {
namespace gpu {
namespace vbd {
namespace tests {

LinearFemMesh::LinearFemMesh(
    Eigen::Ref<GpuMatrixX const> const& V,
    Eigen::Ref<GpuIndexMatrixX const> const& T)
    : mImpl(new fem::Mesh<fem::Tetrahedron<1>, 3>(V.cast<Scalar>(), T.cast<Index>()))
{
}

LinearFemMesh::~LinearFemMesh()
{
    if (mImpl)
        delete mImpl;
}

GpuVectorX LinearFemMesh::QuadratureWeights() const
{
    using MeshType = fem::Mesh<fem::Tetrahedron<1>, 3>;
    auto* mesh     = static_cast<MeshType const*>(mImpl);
    return (fem::DeterminantOfJacobian<1>(*mesh).reshaped().array() / Scalar{6}).cast<GpuScalar>();
}

GpuMatrixX LinearFemMesh::ShapeFunctionGradients() const
{
    using MeshType = fem::Mesh<fem::Tetrahedron<1>, 3>;
    auto* mesh     = static_cast<MeshType const*>(mImpl);
    return fem::ShapeFunctionGradients<1>(*mesh).cast<GpuScalar>();
}

GpuMatrixX LinearFemMesh::LameCoefficients(GpuScalar Y, GpuScalar nu) const
{
    using MeshType = fem::Mesh<fem::Tetrahedron<1>, 3>;
    auto* mesh     = static_cast<MeshType const*>(mImpl);
    auto const [mu, lambda] =
        physics::LameCoefficients(static_cast<Scalar>(Y), static_cast<Scalar>(nu));
    GpuMatrixX lame(2, mesh->E.cols());
    lame.row(0).array() = static_cast<GpuScalar>(mu);
    lame.row(1).array() = static_cast<GpuScalar>(lambda);
    return lame;
}

} // namespace tests
} // namespace vbd
} // namespace gpu
} // namespace pbat