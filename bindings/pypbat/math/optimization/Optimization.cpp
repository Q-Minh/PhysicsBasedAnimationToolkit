#include "Optimization.h"

#include "LineSearch.h"
#include "Newton.h"

namespace pbat::py::math::optimization {

void Bind(nanobind::module_& m)
{
    BindLineSearch(m);
    BindNewton(m);
}

} // namespace pbat::py::math::optimization
