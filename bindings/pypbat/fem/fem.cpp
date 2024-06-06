#include "fem.h"

#include "MassMatrix.h"
#include "Mesh.h"

namespace pbat {
namespace py {
namespace fem {

namespace pyb = pybind11;

void Bind(pyb::module& m)
{
    m.doc() = "Finite Element Method module";
    BindMesh(m);
    BindMassMatrix(m);
}

} // namespace fem
} // namespace py
} // namespace pbat