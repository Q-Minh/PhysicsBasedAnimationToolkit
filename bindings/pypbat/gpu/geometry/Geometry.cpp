#include "Geometry.h"

#include "Aabb.h"
#include "Bodies.h"
#include "Bvh.h"
#include "Points.h"
#include "Simplices.h"
#include "SweepAndPrune.h"

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void Bind(pybind11::module& m)
{
    BindAabb(m);
    BindPoints(m);
    BindSimplices(m);
    BindBodies(m);
    BindSweepAndPrune(m);
    BindBvh(m);
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat