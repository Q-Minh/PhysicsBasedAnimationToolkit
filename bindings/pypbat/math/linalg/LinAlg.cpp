#include "LinAlg.h"

#include "SimplicialLDLT.h"

#include <string>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
    BindSimplicialLDLT(m);
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat