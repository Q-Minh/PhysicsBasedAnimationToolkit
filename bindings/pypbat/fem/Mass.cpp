#include "Mass.h"

#include "Mesh.h"

#include <pbat/fem/Mass.h>
#include <pybind11/eigen.h>
#include <tuple>

namespace pbat {
namespace py {
namespace fem {

void BindMass([[maybe_unused]] pybind11::module& m)
{
    namespace pyb = pybind11;
}

} // namespace fem
} // namespace py
} // namespace pbat