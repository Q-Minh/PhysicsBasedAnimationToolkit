#include "Math.h"

#include "linalg/LinAlg.h"

#include <string>

namespace pbat {
namespace py {
namespace math {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
    auto mlinalg = m.def_submodule("linalg");
    linalg::Bind(mlinalg);
}

} // namespace math
} // namespace py
} // namespace pbat