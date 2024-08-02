#include "Geometry.h"

#include "Points.h"
#include "Simplices.h"
#include "SweepAndPrune.h"

namespace pbat {
namespace py {
namespace gpu {
namespace geometry {

void Bind(pybind11::module& m)
{
    BindPoints(m);
    BindSimplices(m);
    BindSweepAndPrune(m);
}

} // namespace geometry
} // namespace gpu
} // namespace py
} // namespace pbat