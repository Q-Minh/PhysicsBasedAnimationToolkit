#include "Math.h"

#include "MomentFitting.h"
#include "linalg/LinAlg.h"
#include "optimization/Optimization.h"

#include <string>

namespace pbat {
namespace py {
namespace math {

void Bind(nanobind::module_& m)
{
    BindMomentFitting(m);
    auto mlinalg = m.def_submodule("linalg");
    linalg::Bind(mlinalg);
    auto mopt = m.def_submodule("optimization");
    optimization::Bind(mopt);
}

} // namespace math
} // namespace py
} // namespace pbat