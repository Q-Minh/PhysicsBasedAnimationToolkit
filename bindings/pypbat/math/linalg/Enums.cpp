#include "Enums.h"

#include <Eigen/Core>

namespace pbat::py::math::linalg {

void BindEnums(nanobind::module_& m)
{
    namespace nb = nanobind;
    nb::enum_<Eigen::StorageOptions>(m, "StorageOrder")
        .value("RowMajor", Eigen::RowMajor, "Row-major storage order")
        .value("ColMajor", Eigen::ColMajor, "Column-major storage order")
        .export_values();
    nb::enum_<Eigen::UpLoType>(m, "UpLoType")
        .value("Lower", Eigen::Lower, "View matrix as a lower triangular matrix")
        .value("Upper", Eigen::Upper, "View matrix as an upper triangular matrix")
        .value(
            "UnitDiag",
            Eigen::UnitDiag,
            "Matrix has ones on the diagonal; to be used in combination with Lower or Upper")
        .value(
            "ZeroDiag",
            Eigen::ZeroDiag,
            "Matrix has zeros on the diagonal; to be used in combination with Lower or Upper")
        .value(
            "UnitLower",
            Eigen::UnitLower,
            "View matrix as a lower triangular matrix with ones on the diagonal")
        .value(
            "UnitUpper",
            Eigen::UnitUpper,
            "View matrix as an upper triangular matrix with ones on the diagonal")
        .value(
            "StrictlyLower",
            Eigen::StrictlyLower,
            "View matrix as a lower triangular matrix with zeros on the diagonal")
        .value(
            "StrictlyUpper",
            Eigen::StrictlyUpper,
            "View matrix as an upper triangular matrix with zeros on the diagonal")
        .value(
            "SelfAdjoint",
            Eigen::SelfAdjoint,
            "Used in BandMatrix and SelfAdjointView to indicate that the matrix is self-adjoint")
        .value(
            "Symmetric",
            Eigen::Symmetric,
            "Used to support symmetric, non-selfadjoint, complex matrices")
        .export_values();
    nb::enum_<Eigen::ComputationInfo>(m, "ComputationInfo")
        .value("Success", Eigen::Success, "Computation successful")
        .value("NumericalIssue", Eigen::NumericalIssue, "Numerical issue (e.g. non-convergence)")
        .value("NoConvergence", Eigen::NoConvergence, "No convergence")
        .value("InvalidInput", Eigen::InvalidInput, "Invalid input or parameters")
        .export_values();
}

} // namespace pbat::py::math::linalg