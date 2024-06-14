#include "fem.h"

#include "gen/Gradient.h"
#include "gen/HyperElasticPotential.h"
#include "gen/Jacobian.h"
#include "gen/LaplacianMatrix.h"
#include "gen/LoadVector.h"
#include "gen/MassMatrix.h"
#include "gen/Mesh.h"
#include "gen/ShapeFunctions.h"

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