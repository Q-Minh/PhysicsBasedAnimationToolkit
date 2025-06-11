#include "LinAlg.h"

#include "Cholmod.h"
#include "Pardiso.h"
#include "SimplicialLDLT.h"
#include "SparsityPattern.h"

#include <string>

namespace pbat::py::math::linalg {

void Bind(pybind11::module& m)
{
    namespace pyb = pybind11;
    BindCholmod(m);
    BindPardiso(m);
    BindSimplicialLDLT(m);
    BindSparsityPattern(m);
}

} // namespace pbat::py::math::linalg