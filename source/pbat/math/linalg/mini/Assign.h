#ifndef PBAT_MATH_LINALG_MINI_ASSIGN_H
#define PBAT_MATH_LINALG_MINI_ASSIGN_H

#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

#define DefineMatrixMatrixAssign(FunctionName, Operator)                                         \
    template <class TLhsMatrix, class TRhsMatrix>                                                \
    PBAT_HOST_DEVICE void FunctionName(TLhsMatrix&& A, TRhsMatrix&& B)                           \
    {                                                                                            \
        using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;                                   \
        using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;                                   \
        static_assert(CMatrix<LhsMatrixType>, "Left input must satisfy concept CMatrix");        \
        static_assert(CMatrix<RhsMatrixType>, "Right input must satisfy concept CMatrix");       \
        static_assert(                                                                           \
            LhsMatrixType::RowsAtCompileTime == RhsMatrixType::RowsAtCompileTime and             \
                LhsMatrixType::ColsAtCompileTime == RhsMatrixType::ColsAtCompileTime,            \
            "A and B must have same dimensions");                                                \
        if constexpr (LhsMatrixType::IsRowMajor and RhsMatrixType::IsRowMajor)                   \
        {                                                                                        \
            auto fCols = [&]<auto... J>(auto i, std::index_sequence<J...>) {                     \
                ((std::forward<TLhsMatrix>(A)(i, J) Operator std::forward<TRhsMatrix>(B)(i, J)), \
                 ...);                                                                           \
            };                                                                                   \
            auto fRows = [&]<auto... I>(std::index_sequence<I...>) {                             \
                (fCols(I, std::make_index_sequence<LhsMatrixType::ColsAtCompileTime>()), ...);   \
            };                                                                                   \
            fRows(std::make_index_sequence<LhsMatrixType::RowsAtCompileTime>());                 \
        }                                                                                        \
        else                                                                                     \
        {                                                                                        \
            auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {                     \
                ((std::forward<TLhsMatrix>(A)(I, j) Operator std::forward<TRhsMatrix>(B)(I, j)), \
                 ...);                                                                           \
            };                                                                                   \
            auto fCols = [&]<auto... J>(std::index_sequence<J...>) {                             \
                (fRows(J, std::make_index_sequence<LhsMatrixType::RowsAtCompileTime>()), ...);   \
            };                                                                                   \
            fCols(std::make_index_sequence<LhsMatrixType::ColsAtCompileTime>());                 \
        }                                                                                        \
    }

#define DefineMatrixScalarAssign(FunctionName, Operator)                                       \
    template <class TLhsMatrix>                                                                \
    PBAT_HOST_DEVICE void FunctionName(                                                        \
        TLhsMatrix&& A,                                                                        \
        typename std::remove_cvref_t<TLhsMatrix>::Scalar k)                                    \
    {                                                                                          \
        using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;                                 \
        static_assert(CMatrix<LhsMatrixType>, "Left input must satisfy concept CMatrix");      \
        if constexpr (LhsMatrixType::IsRowMajor)                                               \
        {                                                                                      \
            auto fCols = [&]<auto... J>(auto i, std::index_sequence<J...>) {                   \
                ((std::forward<TLhsMatrix>(A)(i, J) Operator k), ...);                         \
            };                                                                                 \
            auto fRows = [&]<auto... I>(std::index_sequence<I...>) {                           \
                (fCols(I, std::make_index_sequence<LhsMatrixType::ColsAtCompileTime>()), ...); \
            };                                                                                 \
            fRows(std::make_index_sequence<LhsMatrixType::RowsAtCompileTime>());               \
        }                                                                                      \
        else                                                                                   \
        {                                                                                      \
            auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {                   \
                ((std::forward<TLhsMatrix>(A)(I, j) Operator k), ...);                         \
            };                                                                                 \
            auto fCols = [&]<auto... J>(std::index_sequence<J...>) {                           \
                (fRows(J, std::make_index_sequence<LhsMatrixType::RowsAtCompileTime>()), ...); \
            };                                                                                 \
            fCols(std::make_index_sequence<LhsMatrixType::ColsAtCompileTime>());               \
        }                                                                                      \
    }

DefineMatrixMatrixAssign(Assign, =);
DefineMatrixMatrixAssign(AddAssign, +=);
DefineMatrixMatrixAssign(SubtractAssign, -=);
DefineMatrixScalarAssign(MultiplyAssign, *=);
DefineMatrixScalarAssign(DivideAssign, /=);

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_ASSIGN_H