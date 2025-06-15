#include "SimplicialLDLT.h"

#include <Eigen/SparseCholesky>
#include <pbat/Aliases.h>
#include <pbat/common/ConstexprFor.h>
#include <pbat/profiling/Profiling.h>
#include <pybind11/eigen.h>
#include <string>
#include <tuple>
#include <type_traits>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

struct SimplicialLdlt
{
    enum class EScalar { Float32, Float64 } eScalar;
    enum class EStorageIndex { Int32, Int64 } eIndex;
    enum class EStorageOrder { Column, Row } eStorageOrder;
    enum class EOrdering { Natural, AMD, COLAMD } eOrdering;
    void* pImpl;

    template <class Func>
    void ForScalarType(Func f)
    {
        if (eScalar == EScalar::Float32)
        {
            f.template operator()<float>();
        }
        else if (eScalar == EScalar::Float64)
        {
            f.template operator()<double>();
        }
    };

    template <class Func>
    void ForIndexType(Func f)
    {
        if (eIndex == EStorageIndex::Int32)
        {
            f.template operator()<std::int32_t>();
        }
        else if (eIndex == EStorageIndex::Int64)
        {
            f.template operator()<std::int64_t>();
        }
    };

    template <class Func>
    void ForOrderingType(Func f)
    {
        if (eOrdering == EOrdering::Natural)
        {
            f.template operator()<Eigen::NaturalOrdering<typename CSCMatrix::StorageIndex>>();
        }
        else if (eOrdering == EOrdering::AMD)
        {
            f.template operator()<Eigen::AMDOrdering<typename CSCMatrix::StorageIndex>>();
        }
        else if (eOrdering == EOrdering::COLAMD)
        {
            f.template operator()<Eigen::COLAMDOrdering<typename CSCMatrix::StorageIndex>>();
        }
    };

    template <class Func>
    void ForStorageOrder(Func f)
    {
        if (eStorageOrder == EStorageOrder::Column)
        {
            f.template operator()<Eigen::ColMajor>();
        }
        else if (eStorageOrder == EStorageOrder::Row)
        {
            f.template operator()<Eigen::RowMajor>();
        }
    };

    template <class Func>
    void Apply(Func f) const
    {
        ForScalarType([&]<class TScalar>() {
            ForIndexType([&]<class TIndex>() {
                ForOrderingType([&]<class TOrdering>() {
                    ForStorageOrder([&]<Eigen::StorageOptions Options>() {
                        using SimplicialLdltType = Eigen::SimplicialLDLT<
                            Eigen::SparseMatrix<TScalar, Options, TIndex>,
                            Eigen::Lower,
                            TOrdering>;
                        SimplicialLdltType* pSimplicialLdlt =
                            static_cast<SimplicialLdltType*>(pImpl);
                        f.template operator()<SimplicialLdltType>(pSimplicialLdlt);
                    });
                });
            });
        });
    }

    SimplicialLdlt(
        EScalar scalar,
        EStorageIndex index,
        EStorageOrder storageOrder,
        EOrdering ordering)
        : eScalar(scalar),
          eIndex(index),
          eStorageOrder(storageOrder),
          eOrdering(ordering),
          pImpl(nullptr)
    {
        Apply([&]<class TSimplicialLdltType>(TSimplicialLdltType* pSimplicialLdlt) {
            pImpl = new TSimplicialLdltType();
        });
    }

    SimplicialLdlt(SimplicialLdlt const&)            = delete;
    SimplicialLdlt& operator=(SimplicialLdlt const&) = delete;
    SimplicialLdlt(SimplicialLdlt&& other) noexcept
        : eScalar(other.eScalar),
          eIndex(other.eIndex),
          eStorageOrder(other.eStorageOrder),
          eOrdering(other.eOrdering),
          pImpl(other.pImpl)
    {
        other.pImpl = nullptr;
    }
    SimplicialLdlt& operator=(SimplicialLdlt&& other) noexcept
    {
        if (this != &other)
        {
            eScalar       = other.eScalar;
            eIndex        = other.eIndex;
            eStorageOrder = other.eStorageOrder;
            eOrdering     = other.eOrdering;
            pImpl         = other.pImpl;
            other.pImpl   = nullptr;
        }
        return *this;
    }

    

    void TryDeallocate()
    {
        if (!pImpl)
            return;
        Apply([&]<class TSimplicialLdltType>(TSimplicialLdltType* pSimplicialLdlt) {
            delete pSimplicialLdlt;
        });
    }

    ~SimplicialLdlt() { TryDeallocate(); }
};

void BindSimplicialLDLT(pybind11::module& m)
{
    namespace pyb = pybind11;

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

            pyb::class_<SimplicialLdltType>(m, className.data())
                .def(pyb::init<>())
                .def(
                    "analyze",
                    [=](SimplicialLdltType& ldlt, SparseMatrixType const& A) {
                        pbat::profiling::Profile(
                            "pbat.math.linalg." + className + ".analyze",
                            [&]() { ldlt.analyzePattern(A); });
                    },
                    pyb::arg("A"))
                .def(
                    "compute",
                    [=](SimplicialLdltType& ldlt,
                        SparseMatrixType const& A) -> SimplicialLdltType& {
                        pbat::profiling::Profile(
                            "pbat.math.linalg." + className + ".compute",
                            [&]() { ldlt.compute(A); });
                        return ldlt;
                    },
                    pyb::arg("A"))
                .def_property_readonly("d", &SimplicialLdltType::vectorD)
                .def_property_readonly("determinant", &SimplicialLdltType::determinant)
                .def(
                    "factorize",
                    [=](SimplicialLdltType& ldlt, SparseMatrixType const& A) {
                        pbat::profiling::Profile(
                            "pbat.math.linalg." + className + ".factorize",
                            [&]() { ldlt.factorize(A); });
                    },
                    pyb::arg("A"))
                .def_property_readonly(
                    "L",
                    [](SimplicialLdltType const& ldlt) -> SparseMatrixType {
                        SparseMatrixType L = ldlt.matrixL();
                        return L;
                    })
                .def_property_readonly(
                    "p",
                    [](SimplicialLdltType const& ldlt) { return ldlt.permutationP().indices(); })
                .def_property_readonly(
                    "pinv",
                    [](SimplicialLdltType const& ldlt) { return ldlt.permutationPinv().indices(); })
                .def_property_readonly(
                    "shape",
                    [](SimplicialLdltType const& ldlt) {
                        return std::make_tuple(ldlt.rows(), ldlt.cols());
                    })
                .def(
                    "shift",
                    [](SimplicialLdltType& ldlt, Scalar offset, Scalar scale) {
                        ldlt.setShift(offset, scale);
                    },
                    pyb::arg("offset"),
                    pyb::arg("scale"))
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
                    pyb::arg("B"))
                .def_property_readonly("status", [](SimplicialLdltType const& ldlt) -> std::string {
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