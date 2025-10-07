#include "Vbd.h"

#include "Integrator.h"

namespace pbat::py::gpu::vbd {

void Bind(nanobind::module_& m)
{
    BindIntegrator(m);
}

} // namespace pbat::py::gpu::vbd