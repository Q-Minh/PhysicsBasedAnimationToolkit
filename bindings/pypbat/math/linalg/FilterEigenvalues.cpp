#include "FilterEigenvalues.h"

#include <nanobind/eigen/dense.h>
#include <pbat/Aliases.h>
#include <pbat/math/linalg/FilterEigenvalues.h>

namespace pbat::py::math::linalg {

void BindFilterEigenvalues(nanobind::module_& m)
{
    namespace nb = nanobind;
    using namespace pbat::math::linalg;
    nb::enum_<EEigenvalueFilter>(m, "EigenvalueFilter")
        .value("NoFilter", EEigenvalueFilter::None)
        .value("SpdProjection", EEigenvalueFilter::SpdProjection)
        .value("FlipNegative", EEigenvalueFilter::FlipNegative)
        .export_values();

    m.def(
        "filter_eigs",
        [](nb::DRef<MatrixX const> A, EEigenvalueFilter mode) -> std::tuple<MatrixX, bool> {
            MatrixX B(A.rows(), A.cols());
            bool result = FilterEigenvalues(A, mode, B);
            return {B, result};
        },
        nb::arg("A"),
        nb::arg("mode") = EEigenvalueFilter::SpdProjection,
        "Make a symmetric matrix positive definite using the specified mode.\n\n"
        "Args\n"
        "    A (numpy.ndarray): `k x k` symmetric matrix.\n"
        "    mode (EigenvalueFilter): Eigenvalue filtering mode (default: SpdProjection).\n\n"
        "Returns\n"
        "    Tuple[numpy.ndarray, bool]: The tuple (B, result) where B is the filtered "
        "matrix and result is true if successful.");
}

} // namespace pbat::py::math::linalg