#include "SimplicialLDLT.h"

#include <Eigen/SparseCholesky>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/stl/tuple.h>
#include <pbat/Aliases.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/profiling/Profiling.h>
#include <string>
#include <tuple>
#include <type_traits>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindSimplicialLDLT(nanobind::module_& m)
{
    namespace nb = nanobind;

    using LdltCscNat = Eigen::SimplicialLDLT<
        CSCMatrix,
        Eigen::Lower,
        Eigen::NaturalOrdering<typename CSCMatrix::StorageIndex>>;
    using LdltCscAmd = Eigen::SimplicialLDLT<
        CSCMatrix,
        Eigen::Lower,
        Eigen::AMDOrdering<typename CSCMatrix::StorageIndex>>;
    using LdltCscColamd = Eigen::SimplicialLDLT<
        CSCMatrix,
        Eigen::Lower,
        Eigen::COLAMDOrdering<typename CSCMatrix::StorageIndex>>;

    using LdltCsrNat = Eigen::SimplicialLDLT<
        CSRMatrix,
        Eigen::Lower,
        Eigen::NaturalOrdering<typename CSRMatrix::StorageIndex>>;
    using LdltCsrAmd = Eigen::SimplicialLDLT<
        CSRMatrix,
        Eigen::Lower,
        Eigen::AMDOrdering<typename CSRMatrix::StorageIndex>>;
    using LdltCsrColamd = Eigen::SimplicialLDLT<
        CSRMatrix,
        Eigen::Lower,
        Eigen::COLAMDOrdering<typename CSRMatrix::StorageIndex>>;

    common::ForTypes<LdltCscNat, LdltCscAmd, LdltCscColamd, LdltCsrNat, LdltCsrAmd, LdltCsrColamd>(
        [&]<class SimplicialLdltType> {
            using SparseMatrixType      = typename SimplicialLdltType::MatrixType;
            std::string const className = []() {
                if constexpr (std::is_same_v<SimplicialLdltType, LdltCscNat>)
                    return "SimplicialLdlt_Csc_Natural";
                if constexpr (std::is_same_v<SimplicialLdltType, LdltCscAmd>)
                    return "SimplicialLdlt_Csc_AMD";
                if constexpr (std::is_same_v<SimplicialLdltType, LdltCscColamd>)
                    return "SimplicialLdlt_Csc_COLAMD";
                if constexpr (std::is_same_v<SimplicialLdltType, LdltCsrNat>)
                    return "SimplicialLdlt_Csr_Natural";
                if constexpr (std::is_same_v<SimplicialLdltType, LdltCsrAmd>)
                    return "SimplicialLdlt_Csr_AMD";
                if constexpr (std::is_same_v<SimplicialLdltType, LdltCsrColamd>)
                    return "SimplicialLdlt_Csr_COLAMD";
            }();

            nb::class_<SimplicialLdltType>(m, className.data())
                .def(nb::init<>())
                .def(
                    "analyze",
                    [=](SimplicialLdltType& ldlt, SparseMatrixType const& A) {
                        pbat::profiling::Profile(
                            "pbat.math.linalg." + className + ".analyze",
                            [&]() { ldlt.analyzePattern(A); });
                    },
                    nb::arg("A"))
                .def(
                    "compute",
                    [=](SimplicialLdltType& ldlt,
                        SparseMatrixType const& A) -> SimplicialLdltType& {
                        pbat::profiling::Profile(
                            "pbat.math.linalg." + className + ".compute",
                            [&]() { ldlt.compute(A); });
                        return ldlt;
                    },
                    nb::arg("A"))
                .def_prop_ro("d", &SimplicialLdltType::vectorD)
                .def_prop_ro("determinant", &SimplicialLdltType::determinant)
                .def(
                    "factorize",
                    [=](SimplicialLdltType& ldlt, SparseMatrixType const& A) {
                        pbat::profiling::Profile(
                            "pbat.math.linalg." + className + ".factorize",
                            [&]() { ldlt.factorize(A); });
                    },
                    nb::arg("A"))
                .def_prop_ro(
                    "L",
                    [](SimplicialLdltType const& ldlt) -> SparseMatrixType {
                        SparseMatrixType L = ldlt.matrixL();
                        return L;
                    })
                .def_prop_ro(
                    "p",
                    [](SimplicialLdltType const& ldlt) { return ldlt.permutationP().indices(); })
                .def_prop_ro(
                    "pinv",
                    [](SimplicialLdltType const& ldlt) { return ldlt.permutationPinv().indices(); })
                .def_prop_ro(
                    "shape",
                    [](SimplicialLdltType const& ldlt) {
                        return std::make_tuple(ldlt.rows(), ldlt.cols());
                    })
                .def(
                    "shift",
                    [](SimplicialLdltType& ldlt, Scalar offset, Scalar scale) {
                        ldlt.setShift(offset, scale);
                    },
                    nb::arg("offset"),
                    nb::arg("scale"))
                .def(
                    "solve",
                    [=](SimplicialLdltType const& ldlt,
                        Eigen::Ref<MatrixX const> const& B) -> MatrixX {
                        return pbat::profiling::Profile(
                            "pbat.math.linalg." + className + ".solve",
                            [&]() {
                                MatrixX X = ldlt.solve(B);
                                return X;
                            });
                    },
                    nb::arg("B"))
                .def_prop_ro("status", [](SimplicialLdltType const& ldlt) -> std::string {
                    Eigen::ComputationInfo const info = ldlt.info();
                    switch (info)
                    {
                        case Eigen::ComputationInfo::Success: return "Success";
                        case Eigen::ComputationInfo::NumericalIssue: return "Numerical issue";
                        case Eigen::ComputationInfo::NoConvergence: return "No convergence";
                        case Eigen::ComputationInfo::InvalidInput: return "Invalid input";
                        default: return "";
                    }
                });
        });
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat