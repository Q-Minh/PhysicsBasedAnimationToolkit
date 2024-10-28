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
            LhsMatrixType::kRows == RhsMatrixType::kRows and                                     \
                LhsMatrixType::kCols == RhsMatrixType::kCols,                                    \
            "A and B must have same dimensions");                                                \
        if constexpr (LhsMatrixType::bRowMajor and RhsMatrixType::bRowMajor)                     \
        {                                                                                        \
            auto fCols = [&]<auto... J>(auto i, std::index_sequence<J...>) {                     \
                ((std::forward<TLhsMatrix>(A)(i, J) Operator std::forward<TRhsMatrix>(B)(i, J)), \
                 ...);                                                                           \
            };                                                                                   \
            auto fRows = [&]<auto... I>(std::index_sequence<I...>) {                             \
                (fCols(I, std::make_index_sequence<LhsMatrixType::kCols>()), ...);               \
            };                                                                                   \
            fRows(std::make_index_sequence<LhsMatrixType::kRows>());                             \
        }                                                                                        \
        else                                                                                     \
        {                                                                                        \
            auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {                     \
                ((std::forward<TLhsMatrix>(A)(I, j) Operator std::forward<TRhsMatrix>(B)(I, j)), \
                 ...);                                                                           \
            };                                                                                   \
            auto fCols = [&]<auto... J>(std::index_sequence<J...>) {                             \
                (fRows(J, std::make_index_sequence<LhsMatrixType::kRows>()), ...);               \
            };                                                                                   \
            fCols(std::make_index_sequence<LhsMatrixType::kCols>());                             \
        }                                                                                        \
    }

#define DefineMatrixScalarAssign(FunctionName, Operator)                                  \
    template <class TLhsMatrix>                                                           \
    PBAT_HOST_DEVICE void FunctionName(                                                   \
        TLhsMatrix&& A,                                                                   \
        typename std::remove_cvref_t<TLhsMatrix>::ScalarType k)                           \
    {                                                                                     \
        using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;                            \
        static_assert(CMatrix<LhsMatrixType>, "Left input must satisfy concept CMatrix"); \
        if constexpr (LhsMatrixType::bRowMajor)                                           \
        {                                                                                 \
            auto fCols = [&]<auto... J>(auto i, std::index_sequence<J...>) {              \
                ((std::forward<TLhsMatrix>(A)(i, J) Operator k), ...);                    \
            };                                                                            \
            auto fRows = [&]<auto... I>(std::index_sequence<I...>) {                      \
                (fCols(I, std::make_index_sequence<LhsMatrixType::kCols>()), ...);        \
            };                                                                            \
            fRows(std::make_index_sequence<LhsMatrixType::kRows>());                      \
        }                                                                                 \
        else                                                                              \
        {                                                                                 \
            auto fRows = [&]<auto... I>(auto j, std::index_sequence<I...>) {              \
                ((std::forward<TLhsMatrix>(A)(I, j) Operator k), ...);                    \
            };                                                                            \
            auto fCols = [&]<auto... J>(std::index_sequence<J...>) {                      \
                (fRows(J, std::make_index_sequence<LhsMatrixType::kRows>()), ...);        \
            };                                                                            \
            fCols(std::make_index_sequence<LhsMatrixType::kCols>());                      \
        }                                                                                 \
    }

DefineMatrixMatrixAssign(Assign, =);
DefineMatrixMatrixAssign(AddAssign, +=);
DefineMatrixMatrixAssign(SubtractAssign, -=);
DefineMatrixScalarAssign(AssignScalar, =);
DefineMatrixScalarAssign(MultiplyAssign, *=);
DefineMatrixScalarAssign(DivideAssign, /=);

#define PBAT_MINI_ASSIGN_API(SelfType)                          \
    template <class TOtherMatrix>                          \
    PBAT_HOST_DEVICE SelfType& operator=(TOtherMatrix&& B) \
    {                                                      \
        Assign(*this, std::forward<TOtherMatrix>(B));      \
        return *this;                                      \
    }

} // namespace mini
} // namespace linalg
} // namespace math
} // namespace pbat

#endif // PBAT_MATH_LINALG_MINI_ASSIGN_H