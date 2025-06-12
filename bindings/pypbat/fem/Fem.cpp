#include "Fem.h"

#include "Gradient.h"
#include "HyperElasticPotential.h"
#include "Jacobian.h"
#include "Laplacian.h"
#include "LoadVector.h"
#include "Mass.h"
#include "Mesh.h"
#include "MeshQuadrature.h"
#include "ShapeFunctions.h"

namespace pbat {
namespace py {
namespace fem {

void Bind(pybind11::module& m)
{
    m.doc() = "Finite Element Method module";
    // Bind mesh first, since all FEM operators depend on it
    BindMesh(m);
    BindGradient(m);
    BindHyperElasticPotential(m);
    BindJacobian(m);
    BindLaplacian(m);
    BindLoadVector(m);
    BindMass(m);
    BindMeshQuadrature(m);
    BindShapeFunctions(m);
}

} // namespace fem
} // namespace py
} // namespace pbat