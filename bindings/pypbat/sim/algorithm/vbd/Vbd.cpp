#include "Vbd.h"

#include "Anderson.h"
#include "Broyden.h"
#include "Chebyshev.h"
#include "Core.h"

namespace pbat::py::sim::algorithm::vbd {

void Bind(nanobind::module_& m)
{
    BindCore(m);
    BindChebyshev(m);
    BindAnderson(m);
    BindBroyden(m);
}

} // namespace pbat::py::sim::algorithm::vbd
