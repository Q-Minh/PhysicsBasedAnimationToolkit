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
    typename TMatrix::Scalar;
    {
        TMatrix::RowsAtCompileTime
    } -> std::convertible_to<int>;
    {
        TMatrix::ColsAtCompileTime
    } -> std::convertible_to<int>;
    {
        TMatrix::IsRowMajor
    } -> std::convertible_to<bool>;
    // WARNING: This constraint causes compile errors with nvcc, I don't know why!
#if not defined(__CUDACC__)
    {
        M(std::declval<int>(), std::declval<int>())
    } -> std::convertible_to<typename TMatrix::Scalar>;
#endif
};

// WARNING: These constraints don't copmile with nvcc, unfortunately. Only use CMatrix concept is
// CUDA code.
#if not defined(__CUDACC__)

template <class TMatrix>
concept CReadableMatrix = CMatrix<TMatrix> && requires(TMatrix M)
{
    {
        M.Rows()
    } -> std::convertible_to<int>;
    {
        M.Cols()
    } -> std::convertible_to<int>;
    {
        M.Transpose()
    } -> CMatrix;
    {
        M.Slice(std::declval<int>(), std::declval<int>())
    } -> CMatrix;
    {
        M.Col(std::declval<int>(), std::declval<int>())
    } -> CMatrix;
    {
        M.Row(std::declval<int>(), std::declval<int>())
    } -> CMatrix;
    {
        M(std::declval<int>(), std::declval<int>())
    } -> std::convertible_to<typename TMatrix::Scalar>;
};

template <class TMatrix>
concept CWriteableMatrix = CMatrix<TMatrix> && requires(TMatrix M)
{
    {M(std::declval<int>(), std::declval<int>()) = std::declval<typename TMatrix::Scalar>()};
    {M = std::declval<TMatrix>()};
};

template <class TMatrix>
concept CReadableVectorizedMatrix = CMatrix<TMatrix> && requires(TMatrix M)
{
    {
        M(std::declval<int>())
    } -> std::convertible_to<typename TMatrix::Scalar>;
    {
        M[std::declval<int>()]
    } -> std::convertible_to<typename TMatrix::Scalar>;
};

template <class TMatrix>
concept CWriteableVectorizedMatrix = CMatrix<TMatrix> && requires(TMatrix M)
{
    {M(std::declval<int>()) = std::declval<typename TMatrix::Scalar>()};
    {M[std::declval<int>()] = std::declval<typename TMatrix::Scalar>()};
};

#endif // not defined(__CUDACC__)

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_CONCEPTS_CUH