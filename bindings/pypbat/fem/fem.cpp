#include "fem.h"

#include "Gradient.h"
#include "HyperElasticPotential.h"
#include "Jacobian.h"
#include "LaplacianMatrix.h"
#include "LoadVector.h"
#include "MassMatrix.h"
#include "Mesh.h"
#include "ShapeFunctions.h"

namespace pbat {
namespace py {
namespace fem {

void Bind(pybind11::module& m)
{
    m.doc() = "Finite Element Method module";
    BindGradient(m);
    BindHyperElasticPotential(m);
    BindJacobian(m);
    BindLaplacianMatrix(m);
    BindLoadVector(m);
    BindMassMatrix(m);
    BindMesh(m);
    BindShapeFunctions(m);
}

} // namespace fem
} // namespace py
} // namespace pbat