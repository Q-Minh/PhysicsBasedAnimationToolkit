#ifndef PBAT_MATH_LINALG_MINI_CONCEPTS_CUH
#define PBAT_MATH_LINALG_MINI_CONCEPTS_CUH

#include <concepts>
#include <type_traits>
#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

template <class TMatrix>
concept CMatrix = requires(TMatrix M)
{
    typename TMatrix::ScalarType;
    {
        TMatrix::kRows
    } -> std::convertible_to<int>;
    {
        TMatrix::kCols
    } -> std::convertible_to<int>;
    {
        TMatrix::bRowMajor
    } -> std::convertible_to<bool>;
    // WARNING: It seems that in our math/linalg/mini library, any concept constraint that uses
    // member functions with the . notation fails. This probably has something to do with M not
    // being a concrete type at the time when we are evaluating the concept constraint.
    //{
    //    M.Rows()
    //} -> std::convertible_to<int>;
    //{
    //    M.Cols()
    //} -> std::convertible_to<int>;
    // WARNING: This constraint causes compile errors with nvcc, I don't know why!
#if not defined(__CUDACC__)
    {
        M(std::declval<int>(), std::declval<int>())
    } -> std::convertible_to<typename TMatrix::ScalarType>;
#endif
};

// WARNING: These constraints don't compile with nvcc, unfortunately. Only use CMatrix concept in
// CUDA code.
#if not defined(__CUDACC__)

template <class TMatrix>
concept CReadableMatrix = CMatrix<TMatrix> and requires(TMatrix M)
{
    true;
    // WARNING:
    // Unfortunately, these constraints cause compilation error, due to ambiguity between the const
    // and non-const versions of these member function calls (tested with MSVC).
    //{M.Transpose()};
    //{M.Slice(std::declval<int>(), std::declval<int>())};
    //{M.Col(std::declval<int>())};
    //{M.Row(std::declval<int>())};
};

template <class TMatrix>
concept CWriteableMatrix = CMatrix<TMatrix> and requires(TMatrix M)
{
    {M(std::declval<int>(), std::declval<int>()) = std::declval<typename TMatrix::ScalarType>()};
    {M = std::declval<TMatrix>()};
};

template <class TMatrix>
concept CReadableVectorizedMatrix = CMatrix<TMatrix> and requires(TMatrix M)
{
    {
        M(std::declval<int>())
    } -> std::convertible_to<typename TMatrix::ScalarType>;
    {
        M[std::declval<int>()]
    } -> std::convertible_to<typename TMatrix::ScalarType>;
};

template <class TMatrix>
concept CWriteableVectorizedMatrix = CMatrix<TMatrix> and requires(TMatrix M)
{
    {M(std::declval<int>()) = std::declval<typename TMatrix::ScalarType>()};
    {M[std::declval<int>()] = std::declval<typename TMatrix::ScalarType>()};
};

#else

template <class TMatrix>
concept CReadableMatrix = CMatrix<TMatrix>;

template <class TMatrix>
concept CWriteableMatrix = CMatrix<TMatrix>;

template <class TMatrix>
concept CReadableVectorizedMatrix = CMatrix<TMatrix>;

template <class TMatrix>
concept CWriteableVectorizedMatrix = CMatrix<TMatrix>;

#endif // not defined(__CUDACC__)

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#define PBAT_MINI_CHECK_CMATRIX(TMatrix) static_assert(CMatrix<TMatrix>, "CMatrix not satisfied");
#define PBAT_MINI_CHECK_CREADABLEMATRIX(TMatrix) \
    static_assert(CReadableMatrix<TMatrix>, "CReadableMatrix not satisfied");
#define PBAT_MINI_CHECK_CWRITEABLEMATRIX(TMatrix) \
    static_assert(CWriteableMatrix<TMatrix>, "CWriteableMatrix not satisfied");
#define PBAT_MINI_CHECK_CREADABLEVECTORIZEDMATRIX(TMatrix) \
    static_assert(CReadableVectorizedMatrix<TMatrix>, "CReadableVectorizedMatrix not satisfied");
#define PBAT_MINI_CHECK_CWRITEABLEVECTORIZEDMATRIX(TMatrix) \
    static_assert(CWriteableVectorizedMatrix<TMatrix>, "CWriteableVectorizedMatrix not satisfied");

#define PBAT_MINI_CHECK_READABLE_CONCEPTS(TMatrix) \
    PBAT_MINI_CHECK_CREADABLEMATRIX(TMatrix);      \
    PBAT_MINI_CHECK_CREADABLEVECTORIZEDMATRIX(TMatrix);

#define PBAT_MINI_CHECK_WRITEABLE_CONCEPTS(TMatrix) \
    PBAT_MINI_CHECK_CWRITEABLEMATRIX(TMatrix);      \
    PBAT_MINI_CHECK_CWRITEABLEVECTORIZEDMATRIX(TMatrix);

#define PBAT_MINI_DIMENSIONS_API                 \
    PBAT_HOST_DEVICE constexpr auto Rows() const \
    {                                            \
        return kRows;                            \
    }                                            \
    PBAT_HOST_DEVICE constexpr auto Cols() const \
    {                                            \
        return kCols;                            \
    }

#endif // PBAT_MATH_LINALG_MINI_CONCEPTS_CUH