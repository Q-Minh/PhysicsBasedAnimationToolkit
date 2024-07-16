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

void BindCholmod(pybind11::module& m)
{
#ifdef PBAT_USE_SUITESPARSE
    namespace pyb = pybind11;

    common::ForTypes<CSCMatrix, CSRMatrix>([&]<class SparseMatrixType>() {
        std::string const className = []() {
            if constexpr (std::is_same_v<SparseMatrixType, CSCMatrix>)
                return "Cholmod_Csc";
            if constexpr (std::is_same_v<SparseMatrixType, CSRMatrix>)
                return "Cholmod_Csr";
            return "";
        }();
        using CholmodType = pbat::math::linalg::Cholmod;
        pyb::class_<CholmodType>(m, className.data())
            .def(
                "analyze",
                [](CholmodType& llt,
                   SparseMatrixType const& A,
                   CholmodType::ESparseStorage storage) {
                    pbat::profiling::Profile("math.linalg.Cholmod.Analyze", [&]() {
                        llt.Analyze(A, storage);
                    });
                },
                pyb::arg("A"),
                pyb::arg("storage") = CholmodType::ESparseStorage::SymmetricLowerTriangular)
            .def(
                "factorize",
                [](CholmodType& llt,
                   SparseMatrixType const& A,
                   CholmodType::ESparseStorage storage) {
                    bool const bFactorized =
                        pbat::profiling::Profile("math.linalg.Cholmod.Factorize", [&]() {
                            return llt.Factorize(A, storage);
                        });
                    return bFactorized;
                },
                pyb::arg("A"),
                pyb::arg("storage") = CholmodType::ESparseStorage::SymmetricLowerTriangular)
            .def(
                "compute",
                [](CholmodType& llt,
                   SparseMatrixType const& A,
                   CholmodType::ESparseStorage storage) {
                    bool const bFactorized =
                        pbat::profiling::Profile("math.linalg.Cholmod.Compute", [&]() {
                            return llt.Compute(A, storage);
                        });
                    return bFactorized;
                },
                pyb::arg("A"),
                pyb::arg("storage") = CholmodType::ESparseStorage::SymmetricLowerTriangular)
            .def(
                "update",
                [](CholmodType& llt,
                   SparseMatrixType const& U,
                   CholmodType::ESparseStorage storage) {
                    bool const bUpdated =
                        pbat::profiling::Profile("math.linalg.Cholmod.Update", [&]() {
                            return llt.Update(U);
                        });
                    return bUpdated;
                },
                pyb::arg("U"),
                pyb::arg("storage") = CholmodType::ESparseStorage::SymmetricLowerTriangular)
            .def(
                "downdate",
                [](CholmodType& llt,
                   SparseMatrixType const& U,
                   CholmodType::ESparseStorage storage) {
                    bool const bDowndated =
                        pbat::profiling::Profile("math.linalg.Cholmod.Downdate", [&]() {
                            return llt.Downdate(U);
                        });
                    return bDowndated;
                },
                pyb::arg("U"),
                pyb::arg("storage") = CholmodType::ESparseStorage::SymmetricLowerTriangular)
            .def(
                "solve",
                [](CholmodType& llt, Eigen::Ref<MatrixX const> const& B) {
                    return pbat::profiling::Profile("math.linalg.Cholmod.Solve", [&]() {
                        MatrixX X = llt.Solve(B);
                        return X;
                    });
                },
                pyb::arg("B"));
    });
#endif // PBAT_USE_SUITESPARSE
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat