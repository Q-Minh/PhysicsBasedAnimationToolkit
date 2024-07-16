#include "LinAlg.h"

#include "Cholmod.h"
#include "Pardiso.h"
#include "SimplicialLDLT.h"

#include <string>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
    BindCholmod(m);
    BindPardiso(m);
    BindSimplicialLDLT(m);
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat