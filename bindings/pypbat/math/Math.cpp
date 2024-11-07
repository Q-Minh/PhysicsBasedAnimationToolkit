#include "Math.h"

#include "MomentFitting.h"
#include "linalg/LinAlg.h"

#include <string>

namespace pbat {
namespace py {
namespace math {

void Bind(pybind11::module& m)
{
    BindMomentFitting(m);
    auto mlinalg = m.def_submodule("linalg");
    linalg::Bind(mlinalg);
}

} // namespace math
} // namespace py
} // namespace pbat