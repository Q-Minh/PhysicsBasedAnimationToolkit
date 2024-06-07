#include "fem.h"

#include "Jacobian.h"
#include "LaplacianMatrix.h"
#include "LoadVector.h"
#include "MassMatrix.h"
#include "Mesh.h"
#include "ShapeFunctions.h"

namespace pbat {
namespace py {
namespace fem {

namespace pyb = pybind11;

void Bind(pyb::module& m)
{
    m.doc() = "Finite Element Method module";
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