#include "LinAlg.h"

#include "Cholmod.h"
#include "Pardiso.h"
#include "SimplicialLDLT.h"
#include "SparsityPattern.h"

#include <string>

namespace pbat::py::math::linalg {

void Bind(nanobind::module_& m)
{
    namespace nb = nanobind;
    BindCholmod(m);
    BindPardiso(m);
    BindSimplicialLDLT(m);
    BindSparsityPattern(m);
}

} // namespace pbat::py::math::linalg