#include "Geometry.h"

#include "Aabb.h"
#include "Bvh.h"
#include "SweepAndPrune.h"

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void Bind(nanobind::module_& m)
{
    BindAabb(m);
    BindSweepAndPrune(m);
    BindBvh(m);
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat