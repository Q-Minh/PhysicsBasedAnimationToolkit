#include "Cholmod.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <pbat/Aliases.h>
#include <pbat/common/ConstexprFor.h>
#ifdef PBAT_USE_SUITESPARSE
#include <pbat/math/linalg/CholmodSupport.h>
#endif // PBAT_USE_SUITESPARSE
#include <pbat/profiling/Profiling.h>
#include <string>
#include <type_traits>

namespace pbat {
namespace py {
namespace math {
namespace linalg {

/**
 * @brief Enum for floating point data types
 */
enum class EFloatDType { float32, float64 };

EFloatDType GetFloatDTypeOrDefault(
    nanobind::object const& dtype,
    EFloatDType const defaultDType = EFloatDType::float64)
{
    namespace nb            = nanobind;
    bool const bDTypeIsNone = dtype.is_none();
    if (not bDTypeIsNone)
    {
        bool const bCanQueryClassName = (nb::hasattr(dtype, "__class__")) and
                                        (nb::hasattr(dtype.attr("__class__"), "__name__"));
        if (not bCanQueryClassName)
        {
            throw std::runtime_error(
                "dtype must be a numpy dtype scalar or object, but could not query class "
                "name");
        }
    }
    if (bDTypeIsNone)
    {
        return defaultDType;
    }
    else
    {
        std::string const dtypeName =
            nb::cast<std::string>(dtype.attr("__class__").attr("__name__"));
        if (dtypeName == "float32" or dtypeName == "Float32DType")
        {
            return EFloatDType::float32;
        }
        else if (dtypeName == "float64" or dtypeName == "Float64DType")
        {
            return EFloatDType::float64;
        }
        else
        {
            throw std::runtime_error(
                "dtype must be a numpy dtype scalar or object, but got " + dtypeName);
        }
    }
};

/**
 * @brief Enum for integer index types
 */
enum class EIntDType { int32, int64 };

EIntDType
GetIntDTypeOrDefault(nanobind::object const& dtype, EIntDType const defaultDType = EIntDType::int64)
{
    namespace nb            = nanobind;
    bool const bDTypeIsNone = dtype.is_none();
    if (not bDTypeIsNone)
    {
        bool const bCanQueryClassName = (nb::hasattr(dtype, "__class__")) and
                                        (nb::hasattr(dtype.attr("__class__"), "__name__"));
        if (not bCanQueryClassName)
        {
            throw std::runtime_error(
                "dtype must be a numpy dtype scalar or object, but could not query class "
                "name");
        }
    }
    if (bDTypeIsNone)
    {
        return defaultDType;
    }
    else
    {
        std::string const dtypeName =
            nb::cast<std::string>(dtype.attr("__class__").attr("__name__"));
        if (dtypeName == "int32" or dtypeName == "Int32DType")
        {
            return EIntDType::int32;
        }
        else if (dtypeName == "int64" or dtypeName == "Int64DType")
        {
            return EIntDType::int64;
        }
        else
        {
            throw std::runtime_error(
                "dtype must be a numpy dtype scalar or object, but got " + dtypeName);
        }
    }
};

#ifdef PBAT_USE_SUITESPARSE
/**
 * @brief Type-erased Cholmod wrapper
 */
struct Cholmod
{
    /**
     * @brief Construct a new Cholmod object
     * @param options Storage options
     * @param uplo Triangular type
     * @param bSingle Whether to use single precision (true) or double precision (false)
     * @param bLongIndices Whether to use long indices (true) or int indices (false)
     */
    Cholmod(
        Eigen::StorageOptions options = Eigen::ColMajor,
        Eigen::UpLoType uplo          = Eigen::Lower,
        EFloatDType eFloat            = EFloatDType::float64,
        EIntDType eInt                = EIntDType::int64)
        : mOptions(options), mUplo(uplo), mFloat(eFloat), mInt(eInt), mImpl(nullptr)
    {
        Construct();
    }
    Cholmod(Cholmod const&)                = delete;
    Cholmod(Cholmod&&) noexcept            = default;
    Cholmod& operator=(Cholmod const&)     = delete;
    Cholmod& operator=(Cholmod&&) noexcept = default;
    /**
     * @brief Apply a function on the implementation
     * @tparam Func Function type
     * @param f Function to apply
     */
    template <class Func>
    void ApplyOnImpl(Func f)
    {
        pbat::common::ForValues<Eigen::ColMajor, Eigen::RowMajor>(
            [&]<Eigen::StorageOptions options>() {
                pbat::common::ForValues<Eigen::Lower, Eigen::Upper>([&]<Eigen::UpLoType uplo>() {
                    pbat::common::ForValues<EIntDType::int32, EIntDType::int64>(
                        [&]<EIntDType eInt>() {
                            pbat::common::ForValues<EFloatDType::float32, EFloatDType::float64>(
                                [&]<EFloatDType eFloat>() {
                                    if (options == mOptions and uplo == mUplo and eInt == mInt and
                                        eFloat == mFloat)
                                    {
                                        using SparseMatrixType = Eigen::SparseMatrix<
                                            typename std::conditional<
                                                eFloat == EFloatDType::float32,
                                                float,
                                                double>::type,
                                            options,
                                            typename std::conditional<
                                                eInt == EIntDType::int64,
                                                std::int64_t,
                                                std::int32_t>::type>;
                                        using CholmodType =
                                            Eigen::CholmodDecomposition<SparseMatrixType, uplo>;
                                        f(static_cast<CholmodType*>(mImpl));
                                    }
                                });
                        });
                });
            });
    }
    /**
     * @brief Construct the Cholmod object
     */
    void Construct()
    {
        Destroy();
        ApplyOnImpl([this]<typename T>([[maybe_unused]] T* impl) { mImpl = new T(); });
    }
    /**
     * @brief Solve the linear system AX = B
     *
     * @tparam TDerivedB Type of rhs
     * @tparam TDerivedX Type of solution
     * @param B Right-hand side matrix
     * @param X Solution matrix
     */
    template <class TDerivedB, class TDerivedX>
    void Solve(Eigen::MatrixBase<TDerivedB> const& B, Eigen::MatrixBase<TDerivedX>& X)
    {
        ApplyOnImpl([&]<typename T>(T* impl) {
            X = impl->solve(B.cast<typename T::Scalar>()).cast<typename TDerivedX::Scalar>();
        });
    }
    /**
     * @brief Symbolic factorization of A
     *
     * @tparam TDerivedA Type of input matrix
     * @param A The input compressed sparse column self-adjoint matrix
     * @return Computation info
     */
    template <class TDerivedA>
    Eigen::ComputationInfo Analyze(Eigen::SparseMatrixBase<TDerivedA> const& A)
    {
        Eigen::ComputationInfo info{Eigen::InvalidInput};
        ApplyOnImpl([&]<typename T>(T* impl) {
            using CurrentCholmodType = std::remove_const_t<T>;
            using TargetCholmodType  = Eigen::CholmodDecomposition<
                 Eigen::SparseMatrix<
                     typename Eigen::SparseMatrixBase<TDerivedA>::Scalar,
                    Eigen::SparseMatrixBase<TDerivedA>::IsRowMajor ? Eigen::RowMajor :
                                                                      Eigen::ColMajor,
                     typename Eigen::SparseMatrixBase<TDerivedA>::StorageIndex>,
                 CurrentCholmodType::UpLo>;
            if constexpr (std::is_same_v<TargetCholmodType, CurrentCholmodType>)
            {
                impl->analyzePattern(A);
                info = impl->info();
            }
        });
        return info;
    }
    /**
     * @brief Numeric factorization of A
     *
     * @tparam TDerivedA Type of input matrix
     * @param A The input compressed sparse column self-adjoint matrix
     * @return Computation info
     */
    template <class TDerivedA>
    Eigen::ComputationInfo Factorize(Eigen::SparseMatrixBase<TDerivedA> const& A)
    {
        Eigen::ComputationInfo info{Eigen::InvalidInput};
        ApplyOnImpl([&]<typename T>(T* impl) {
            using CurrentCholmodType = std::remove_const_t<T>;
            using TargetCholmodType  = Eigen::CholmodDecomposition<
                 Eigen::SparseMatrix<
                     typename Eigen::SparseMatrixBase<TDerivedA>::Scalar,
                    Eigen::SparseMatrixBase<TDerivedA>::IsRowMajor ? Eigen::RowMajor :
                                                                      Eigen::ColMajor,
                     typename Eigen::SparseMatrixBase<TDerivedA>::StorageIndex>,
                 CurrentCholmodType::UpLo>;
            if constexpr (std::is_same_v<TargetCholmodType, CurrentCholmodType>)
            {
                impl->factorize(A);
                info = impl->info();
            }
        });
        return info;
    }
    /**
     * @brief Symbolic and numeric factorization of A
     *
     * @tparam TDerivedA Type of input matrix
     * @param A The input compressed sparse column self-adjoint matrix
     * @return Computation info
     */
    template <class TDerivedA>
    Eigen::ComputationInfo Compute(Eigen::SparseMatrixBase<TDerivedA> const& A)
    {
        Eigen::ComputationInfo info{Eigen::InvalidInput};
        ApplyOnImpl([&]<typename T>(T* impl) {
            using CurrentCholmodType = std::remove_const_t<T>;
            using TargetCholmodType  = Eigen::CholmodDecomposition<
                 Eigen::SparseMatrix<
                     typename Eigen::SparseMatrixBase<TDerivedA>::Scalar,
                    Eigen::SparseMatrixBase<TDerivedA>::IsRowMajor ? Eigen::RowMajor :
                                                                      Eigen::ColMajor,
                     typename Eigen::SparseMatrixBase<TDerivedA>::StorageIndex>,
                 CurrentCholmodType::UpLo>;
            if constexpr (std::is_same_v<TargetCholmodType, CurrentCholmodType>)
            {
                impl->compute(A);
                info = impl->info();
            }
        });
        return info;
    }
    /**
     * @brief Destroy the Cholmod object
     */
    void Destroy()
    {
        if (mImpl)
        {
            ApplyOnImpl([]<typename T>(T* impl) { delete impl; });
            mImpl = nullptr;
        }
    }
    /**
     * @brief Destroy the Cholmod object
     */
    ~Cholmod() { Destroy(); }

    Eigen::StorageOptions mOptions; ///< Storage order
    Eigen::UpLoType mUplo;          ///< Triangle type
    EFloatDType mFloat; ///< Whether to use single precision (true) or double precision (false)
    EIntDType mInt;     ///< Whether to use long indices (true) or int indices (false)
    void* mImpl;        ///< Pointer to the actual implementation
};
#endif // PBAT_USE_SUITESPARSE

void BindCholmod([[maybe_unused]] nanobind::module_& m)
{
#ifdef PBAT_USE_SUITESPARSE
    namespace nb = nanobind;
    // ERROR:
    // ImportError bad cast when importing _pbat from python due to this Cholmod binding.
    // nb::class_<Cholmod> chol(m, "Cholmod");
    // chol.def(
    //         nb::init<Eigen::StorageOptions, Eigen::UpLoType, EFloatDType, EIntDType>(),
    //         nb::arg("options")     = Eigen::ColMajor,
    //         nb::arg("uplo")        = Eigen::Lower,
    //         nb::arg("coeff_dtype") = EFloatDType::float64,
    //         nb::arg("index_dtype") = EIntDType::int64,
    //         "Cholmod constructor\n\n"
    //         "Args\n"
    //         "    options (int, optional): Storage order, either ColMajor or RowMajor. Defaults "
    //         "to "
    //         "ColMajor.\n"
    //         "    uplo (int, optional): Triangle type, either Lower (0) or Upper (1). Defaults "
    //         "to "
    //         "Lower.\n"
    //         "    coeff_dtype (numpy.dtype, optional): Type of float to use for matrix "
    //         "coefficients. Defaults to numpy.float64.\n"
    //         "    index_dtype (numpy.dtype, optional): Type of int to use for matrix indices. "
    //         "Defaults to numpy.int64.\n")
    //     .def(
    //         "solve",
    //         [](Cholmod& chol, Eigen::Ref<MatrixX const> const& B) {
    //             return pbat::profiling::Profile("pbat.math.linalg.Cholmod.Solve", [&]() {
    //                 MatrixX X;
    //                 chol.Solve(B, X);
    //                 return X;
    //             });
    //         },
    //         nb::arg("B"),
    //         "Solve the linear system AX = B using the current factorization\n\n"
    //         "Args\n"
    //         "    B (numpy.ndarray): Right-hand side matrix\n"
    //         "Returns\n"
    //         "    numpy.ndarray: Solution matrix X\n")
    //     .def(
    //         "solve",
    //         [](Cholmod& chol, Eigen::Ref<VectorX const> const& b) {
    //             return pbat::profiling::Profile("pbat.math.linalg.Cholmod.Solve", [&]() {
    //                 VectorX X;
    //                 chol.Solve(b, X);
    //                 return X;
    //             });
    //         },
    //         nb::arg("b"),
    //         "Solve the linear system Ax = b using the current factorization\n\n"
    //         "Args\n"
    //         "    b (numpy.ndarray): Right-hand side vector\n"
    //         "Returns\n"
    //         "    numpy.ndarray: Solution vector x\n");
    // pbat::common::ForTypes<float, double>([&]<class T>() {
    //     pbat::common::ForValues<Eigen::RowMajor, Eigen::ColMajor>(
    //         [&]<Eigen::StorageOptions options>() {
    //             pbat::common::ForTypes<std::int32_t, std::int64_t>([&]<typename U>() {
    //                 using SparseMatrixType = Eigen::SparseMatrix<T, options, U>;
    //                 chol.def(
    //                         "analyze",
    //                         [](Cholmod& chol, SparseMatrixType const& A) {
    //                             return pbat::profiling::Profile(
    //                                 "pbat.math.linalg.Cholmod.Analyze",
    //                                 [&]() {
    //                                     if (chol.mFloat == EFloatDType::float32)
    //                                         return chol.Analyze(A.cast<float>());
    //                                     else
    //                                         return chol.Analyze(A.cast<double>());
    //                                 });
    //                         },
    //                         nb::arg("A"),
    //                         "Symbolic factorize A\n\n"
    //                         "Args\n"
    //                         "    A (scipy.sparse.csc_matrix): The input compressed sparse column
    //                         " "self-adjoint " "matrix\n" "Returns\n" "    ComputationInfo: Result
    //                         of symbolic factorization\n")
    //                     .def(
    //                         "factorize",
    //                         [](Cholmod& chol, SparseMatrixType const& A) {
    //                             return pbat::profiling::Profile(
    //                                 "pbat.math.linalg.Cholmod.Factorize",
    //                                 [&]() {
    //                                     if (chol.mFloat == EFloatDType::float32)
    //                                         return chol.Factorize(A.cast<float>());
    //                                     else
    //                                         return chol.Factorize(A.cast<double>());
    //                                 });
    //                         },
    //                         nb::arg("A"),
    //                         "Compute the numeric factorization of A\n\n"
    //                         "Args\n"
    //                         "    A (scipy.sparse.csc_matrix): The input compressed sparse column
    //                         " "self-adjoint " "matrix\n" "Returns\n" "    bool: True if the
    //                         factorization was successful, False otherwise\n")
    //                     .def(
    //                         "compute",
    //                         [](Cholmod& chol, SparseMatrixType const& A) {
    //                             return pbat::profiling::Profile(
    //                                 "pbat.math.linalg.Cholmod.Compute",
    //                                 [&]() {
    //                                     if (chol.mFloat == EFloatDType::float32)
    //                                         return chol.Compute(A.cast<float>());
    //                                     else
    //                                         return chol.Compute(A.cast<double>());
    //                                 });
    //                         },
    //                         nb::arg("A"),
    //                         "Compute the symbolic and numeric factorization of A\n\n"
    //                         "Args\n"
    //                         "    A (scipy.sparse.csc_matrix): The input compressed sparse column
    //                         " "self-adjoint " "matrix\n" "Returns\n" "    bool: True if the
    //                         computation was successful, False otherwise\n");
    //             });
    //         });
    // });
#endif // PBAT_USE_SUITESPARSE
}

} // namespace linalg
} // namespace math
} // namespace py
} // namespace pbat