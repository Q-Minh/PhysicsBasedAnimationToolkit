#include "Optimization.h"

#include "LineSearch.h"
#include "Newton.h"
#include "TriangleConstrainedTrustRegionSr1.h"

namespace pbat::py::math::optimization {

void Bind(nanobind::module_& m)
{
    BindLineSearch(m);
    BindNewton(m);
    BindTriangleConstrainedTrustRegionSr1(m);
}

} // namespace pbat::py::math::optimization
