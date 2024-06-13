#include "fem.h"

#include "pbatautogen/Gradient.h"
#include "pbatautogen/HyperElasticPotential.h"
#include "pbatautogen/Jacobian.h"
#include "pbatautogen/LaplacianMatrix.h"
#include "pbatautogen/LoadVector.h"
#include "pbatautogen/MassMatrix.h"
#include "pbatautogen/Mesh.h"
#include "pbatautogen/ShapeFunctions.h"

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