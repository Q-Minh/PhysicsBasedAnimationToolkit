#include "Cholmod.h"

#include <pbat/Aliases.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/math/linalg/Cholmod.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>
#include <string>
#include <type_traits>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

void BindCholmod([[maybe_unused]] pybind11::module& m)
{
#ifdef PBAT_USE_SUITESPARSE
    namespace pyb = pybind11;

    std::string const className = "Cholmod";
    using CholmodType           = pbat::math::linalg::Cholmod;
    pyb::class_<CholmodType> chol(m, className.data());
    chol.def(pyb::init<>());

    pyb::enum_<CholmodType::ESparseStorage>(chol, "SparseStorage")
        .value("SymmetricLowerTriangular", CholmodType::ESparseStorage::SymmetricLowerTriangular)
        .value("SymmetricUpperTriangular", CholmodType::ESparseStorage::SymmetricUpperTriangular)
        .value("Unsymmetric", CholmodType::ESparseStorage::Unsymmetric)
        .export_values();

    chol.def(
        "solve",
        [](CholmodType& llt, Eigen::Ref<MatrixX const> const& B) {
            return pbat::profiling::Profile("pbat.math.linalg.Cholmod.Solve", [&]() {
                MatrixX X = llt.Solve(B);
                return X;
            });
        },
        pyb::arg("B"));

    chol.doc() =
        "Cholmod Cholesky or Bunch-Kaufmann decompositions of matrix A, stored in compressed "
        "sparse column format. If A is stored in compressed row format, then its transpose is "
        "decomposed instead. If A is unsymmetric, then A AT is decomposed (or AT A if A is stored "
        "in CSR format).";

    common::ForTypes<CSCMatrix, CSRMatrix>([&]<class SparseMatrixType>() {
        chol.def(
                "analyze",
                [](CholmodType& llt,
                   SparseMatrixType const& A,
                   CholmodType::ESparseStorage storage) {
                    pbat::profiling::Profile("pbat.math.linalg.Cholmod.Analyze", [&]() {
                        llt.Analyze(A, storage);
                    });
                },
                pyb::arg("A"),
                pyb::arg("storage"))
            .def(
                "factorize",
                [](CholmodType& llt,
                   SparseMatrixType const& A,
                   CholmodType::ESparseStorage storage) {
                    bool const bFactorized =
                        pbat::profiling::Profile("pbat.math.linalg.Cholmod.Factorize", [&]() {
                            return llt.Factorize(A, storage);
                        });
                    return bFactorized;
                },
                pyb::arg("A"),
                pyb::arg("storage"))
            .def(
                "compute",
                [](CholmodType& llt,
                   SparseMatrixType const& A,
                   CholmodType::ESparseStorage storage) {
                    bool const bFactorized =
                        pbat::profiling::Profile("pbat.math.linalg.Cholmod.Compute", [&]() {
                            return llt.Compute(A, storage);
                        });
                    return bFactorized;
                },
                pyb::arg("A"),
                pyb::arg("storage"))
            .def(
                "update",
                [](CholmodType& llt, SparseMatrixType const& U) {
                    bool const bUpdated =
                        pbat::profiling::Profile("pbat.math.linalg.Cholmod.Update", [&]() {
                            return llt.Update(U);
                        });
                    return bUpdated;
                },
                pyb::arg("U"))
            .def(
                "downdate",
                [](CholmodType& llt, SparseMatrixType const& U) {
                    bool const bDowndated =
                        pbat::profiling::Profile("pbat.math.linalg.Cholmod.Downdate", [&]() {
                            return llt.Downdate(U);
                        });
                    return bDowndated;
                },
                pyb::arg("U"));
    });
#endif // PBAT_USE_SUITESPARSE
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat