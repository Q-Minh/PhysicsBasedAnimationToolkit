#include "Common.h"

#include "Buffer.h"

namespace pbat {
namespace py {
namespace gpu {
namespace common {

void Bind(nanobind::module_& m)
{
    BindBuffer(m);
}

} // namespace common
} // namespace gpu
} // namespace py
} // namespace pbat