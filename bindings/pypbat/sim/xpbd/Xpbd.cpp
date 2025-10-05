#include "Xpbd.h"

#include "Data.h"
#include "Integrator.h"

namespace pbat {
namespace py {
namespace sim {
namespace xpbd {

void Bind(nanobind::module_& m)
{
    namespace nb = nanobind;
    BindData(m);
    BindIntegrator(m);
}

} // namespace xpbd
} // namespace sim
} // namespace py
} // namespace pbat