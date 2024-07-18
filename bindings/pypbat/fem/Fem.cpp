#include "Fem.h"

#include "Gradient.h"
#include "Jacobian.h"
#include "Laplacian.h"
#include "MassMatrix.h"
#include "Mesh.h"
#include "ShapeFunctions.h"

namespace pbat {
namespace py {
namespace fem {

void Bind(pybind11::module& m)
{
    m.doc() = "Finite Element Method module";
    BindMesh(m);
}

} // namespace fem
} // namespace py
} // namespace pbat