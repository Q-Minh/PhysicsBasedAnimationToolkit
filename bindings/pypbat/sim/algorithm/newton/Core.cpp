#include "Core.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <pbat/Aliases.h>
#include <pbat/sim/algorithm/newton/Core.h>

namespace pbat::py::sim::algorithm::newton {

void BindCore(nanobind::module_& m)
{
    namespace nb = nanobind;
    using pbat::sim::algorithm::newton::Params;

    nb::class_<Params>(m, "Params")
        .def(nb::init<>(), "Newton solver parameters and buffers.")
        .def_rw("newton", &Params::newton, "Underlying Newton optimizer (math.optimization.Newton)")
        .def_rw(
            "ordering",
            &Params::ordering,
            "Triplet ordering for sparse Hessian assembly (|# triplets| x 1 integer array)")
        .def_rw(
            "hessian",
            &Params::hessian,
            "Sparse Hessian matrix (Eigen::SparseMatrix in CSC format)")
        .def_prop_ro(
            "triplets",
            [](Params const& self) {
                Eigen::Vector<pbat::Index, Eigen::Dynamic> rows(self.triplets.size());
                Eigen::Vector<pbat::Index, Eigen::Dynamic> cols(self.triplets.size());
                Eigen::Vector<pbat::Scalar, Eigen::Dynamic> vals(self.triplets.size());
                for (size_t i = 0; i < self.triplets.size(); ++i)
                {
                    rows(static_cast<pbat::Index>(i)) = self.triplets[i].row();
                    cols(static_cast<pbat::Index>(i)) = self.triplets[i].col();
                    vals(static_cast<pbat::Index>(i)) = self.triplets[i].value();
                }
                return std::make_tuple(rows, cols, vals);
            },
            "Hessian triplets as (rows, cols, vals) arrays.");
}

} // namespace pbat::py::sim::algorithm::newton
