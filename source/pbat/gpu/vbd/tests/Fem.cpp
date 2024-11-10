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
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& T)
    : mImpl(new fem::Mesh<fem::Tetrahedron<1>, 3>(V, T))
{
}

LinearFemMesh::~LinearFemMesh()
{
    if (mImpl)
    {
        using MeshType = fem::Mesh<fem::Tetrahedron<1>, 3>;
        auto* ptr      = static_cast<MeshType*>(mImpl);
        delete ptr;
    }
}

VectorX LinearFemMesh::QuadratureWeights() const
{
    using MeshType = fem::Mesh<fem::Tetrahedron<1>, 3>;
    auto* mesh     = static_cast<MeshType const*>(mImpl);
    return (fem::DeterminantOfJacobian<1>(*mesh).reshaped().array() / Scalar{6});
}

MatrixX LinearFemMesh::ShapeFunctionGradients() const
{
    using MeshType = fem::Mesh<fem::Tetrahedron<1>, 3>;
    auto* mesh     = static_cast<MeshType const*>(mImpl);
    return fem::ShapeFunctionGradients<1>(*mesh);
}

MatrixX LinearFemMesh::LameCoefficients(Scalar Y, Scalar nu) const
{
    using MeshType          = fem::Mesh<fem::Tetrahedron<1>, 3>;
    auto* mesh              = static_cast<MeshType const*>(mImpl);
    auto const [mu, lambda] = physics::LameCoefficients(Y, nu);
    MatrixX lame(2, mesh->E.cols());
    lame.row(0).array() = mu;
    lame.row(1).array() = lambda;
    return lame;
}

} // namespace tests
} // namespace vbd
} // namespace gpu
} // namespace pbat