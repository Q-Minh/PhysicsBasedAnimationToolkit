#ifndef PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_H
#define PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_H

#include <pybind11/pybind11.h>

namespace pbat {
namespace py {
namespace fem {

void BindHyperElasticPotential(pybind11::module& m);

} // namespace fem
} // namespace py
} // namespace pbat

#endif // PYPBAT_FEM_HYPER_ELASTIC_POTENTIAL_H