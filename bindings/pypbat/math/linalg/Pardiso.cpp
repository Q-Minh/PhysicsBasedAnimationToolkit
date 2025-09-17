#include "Pardiso.h"

#ifdef PBAT_USE_INTEL_MKL
    #include <cstdint>
    // NOTE: The default Eigen::SparseMatrix::StorageIndex is int, see
    // Eigen/src/SparseCore/SparseUtil.h line 52.
    #define MKL_INT int
    #include <Eigen/PardisoSupport>
    #include <nanobind/eigen/dense.h>
    #include <nanobind/eigen/sparse.h>
    #include <pbat/Aliases.h>
    #include <pbat/common/ConstexprFor.h>
    #include <pbat/profiling/Profiling.h>
    #include <string>
    #include <tuple>
    #include <type_traits>
#endif // PBAT_USE_INTEL_MKL

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindPardiso([[maybe_unused]] nanobind::module_& m)
{
    namespace nb = nanobind;
#ifdef PBAT_USE_INTEL_MKL
    using PardisoLuCsc   = Eigen::PardisoLU<CSCMatrix>;
    using PardisoLuCsr   = Eigen::PardisoLU<CSRMatrix>;
    using PardisoLltCsc  = Eigen::PardisoLLT<CSCMatrix>;
    using PardisoLltCsr  = Eigen::PardisoLLT<CSRMatrix>;
    using PardisoLdltCsc = Eigen::PardisoLDLT<CSCMatrix>;
    using PardisoLdltCsr = Eigen::PardisoLDLT<CSRMatrix>;

    common::ForTypes<
        PardisoLuCsc,
        PardisoLuCsr,
        PardisoLltCsc,
        PardisoLltCsr,
        PardisoLdltCsc,
        PardisoLdltCsr>([&]<class SolverType>() {
        using SparseMatrixType      = typename SolverType::MatrixType;
        std::string const className = []() {
            if constexpr (std::is_same_v<SolverType, PardisoLuCsc>)
                return "PardisoLU_Csc";
            if constexpr (std::is_same_v<SolverType, PardisoLuCsr>)
                return "PardisoLU_Csr";
            if constexpr (std::is_same_v<SolverType, PardisoLltCsc>)
                return "PardisoLLT_Csc";
            if constexpr (std::is_same_v<SolverType, PardisoLltCsr>)
                return "PardisoLLT_Csr";
            if constexpr (std::is_same_v<SolverType, PardisoLdltCsc>)
                return "PardisoLDLT_Csc";
            if constexpr (std::is_same_v<SolverType, PardisoLdltCsr>)
                return "PardisoLDLT_Csr";
        }();
        using PardisoParameterArrayType =
            std::remove_cvref_t<decltype(std::declval<SolverType>().pardisoParameterArray())>;

        nb::class_<SolverType>(m, className.data())
            .def(nb::init<>())
            .def(
                "analyze",
                [=](SolverType& solver, SparseMatrixType const& A) {
                    pbat::profiling::Profile("pbat.math.linalg." + className + ".analyze", [&]() {
                        solver.analyzePattern(A);
                    });
                },
                nb::arg("A"))
            .def(
                "compute",
                [=](SolverType& solver, SparseMatrixType const& A) -> SolverType& {
                    pbat::profiling::Profile("pbat.math.linalg." + className + ".compute", [&]() {
                        solver.compute(A);
                    });
                    return solver;
                },
                nb::arg("A"))
            .def(
                "factorize",
                [=](SolverType& solver, SparseMatrixType const& A) {
                    pbat::profiling::Profile("pbat.math.linalg." + className + ".factorize", [&]() {
                        solver.factorize(A);
                    });
                },
                nb::arg("A"))
            .def_prop_ro(
                "shape",
                [](SolverType const& solver) {
                    return std::make_tuple(solver.rows(), solver.cols());
                })
            .def(
                "solve",
                [=](SolverType const& solver, Eigen::Ref<MatrixX const> const& B) -> MatrixX {
                    return pbat::profiling::Profile(
                        "pbat.math.linalg." + className + ".solve",
                        [&]() {
                            MatrixX X = solver.solve(B);
                            return X;
                        });
                },
                nb::arg("B"))
            .def_prop_ro(
                "status",
                [](SolverType const& solver) -> std::string {
                    Eigen::ComputationInfo const info = solver.info();
                    switch (info)
                    {
                        case Eigen::ComputationInfo::Success: return "Success";
                        case Eigen::ComputationInfo::NumericalIssue: return "Numerical issue";
                        case Eigen::ComputationInfo::NoConvergence: return "No convergence";
                        case Eigen::ComputationInfo::InvalidInput: return "Invalid input";
                        default: return "";
                    }
                })
            .def_prop_rw(
                "iparm",
                [](SolverType& solver) -> PardisoParameterArrayType const& {
                    return solver.pardisoParameterArray();
                },
                [](SolverType& solver, PardisoParameterArrayType const& iparm) {
                    solver.pardisoParameterArray() = iparm;
                });
    });
#endif // PBAT_USE_INTEL_MKL
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat