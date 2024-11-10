#ifndef PBAT_MATH_LINALG_MINI_ASSIGN_H
#define PBAT_MATH_LINALG_MINI_ASSIGN_H

#include "Concepts.h"
#include "pbat/HostDevice.h"

#include <utility>

namespace pbat {
namespace math {
namespace linalg {
namespace mini {

#define DefineMatrixMatrixAssign(FunctionName, Operator)                                          \
    template <class TLhsMatrix, class TRhsMatrix>                                                 \
    PBAT_HOST_DEVICE void FunctionName(TLhsMatrix&& A, TRhsMatrix&& B)                            \
    {                                                                                             \
        using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;                                    \
        using RhsMatrixType = std::remove_cvref_t<TRhsMatrix>;                                    \
        static_assert(CMatrix<LhsMatrixType>, "Left input must satisfy concept CMatrix");         \
        static_assert(CMatrix<RhsMatrixType>, "Right input must satisfy concept CMatrix");        \
        static_assert(                                                                            \
            LhsMatrixType::kRows == RhsMatrixType::kRows and                                      \
                LhsMatrixType::kCols == RhsMatrixType::kCols,                                     \
            "A and B must have same dimensions");                                                 \
        using IntegerType = std::remove_const_t<decltype(LhsMatrixType::kRows)>;                  \
        if constexpr (LhsMatrixType::bRowMajor and RhsMatrixType::bRowMajor)                      \
        {                                                                                         \
            auto fCols = [&]<IntegerType... J>(                                                   \
                             IntegerType i,                                                       \
                             std::integer_sequence<IntegerType, J...>) {                          \
                ((std::forward<TLhsMatrix>(A)(i, J) Operator std::forward<TRhsMatrix>(B)(i, J)),  \
                 ...);                                                                            \
            };                                                                                    \
            auto fRows = [&]<IntegerType... I>(std::integer_sequence<IntegerType, I...>) {        \
                (fCols(I, std::make_integer_sequence<IntegerType, LhsMatrixType::kCols>()), ...); \
            };                                                                                    \
            fRows(std::make_integer_sequence<IntegerType, LhsMatrixType::kRows>());               \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            auto fRows = [&]<IntegerType... I>(                                                   \
                             IntegerType j,                                                       \
                             std::integer_sequence<IntegerType, I...>) {                          \
                ((std::forward<TLhsMatrix>(A)(I, j) Operator std::forward<TRhsMatrix>(B)(I, j)),  \
                 ...);                                                                            \
            };                                                                                    \
            auto fCols = [&]<IntegerType... J>(std::integer_sequence<IntegerType, J...>) {        \
                (fRows(J, std::make_integer_sequence<IntegerType, LhsMatrixType::kRows>()), ...); \
            };                                                                                    \
            fCols(std::make_integer_sequence<IntegerType, LhsMatrixType::kCols>());               \
        }                                                                                         \
    }

#define DefineMatrixScalarAssign(FunctionName, Operator)                                          \
    template <class TLhsMatrix>                                                                   \
    PBAT_HOST_DEVICE void FunctionName(                                                           \
        TLhsMatrix&& A,                                                                           \
        typename std::remove_cvref_t<TLhsMatrix>::ScalarType k)                                   \
    {                                                                                             \
        using LhsMatrixType = std::remove_cvref_t<TLhsMatrix>;                                    \
        static_assert(CMatrix<LhsMatrixType>, "Left input must satisfy concept CMatrix");         \
        using IntegerType = std::remove_const_t<decltype(LhsMatrixType::kRows)>;                  \
        if constexpr (LhsMatrixType::bRowMajor)                                                   \
        {                                                                                         \
            auto fCols =                                                                          \
                [&]<IntegerType... J>(IntegerType i, std::integer_sequence<IntegerType, J...>) {  \
                    ((std::forward<TLhsMatrix>(A)(i, J) Operator k), ...);                        \
                };                                                                                \
            auto fRows = [&]<IntegerType... I>(std::integer_sequence<IntegerType, I...>) {        \
                (fCols(I, std::make_integer_sequence<IntegerType, LhsMatrixType::kCols>()), ...); \
            };                                                                                    \
            fRows(std::make_integer_sequence<IntegerType, LhsMatrixType::kRows>());               \
        }                                                                                         \
        else                                                                                      \
        {                                                                                         \
            auto fRows =                                                                          \
                [&]<IntegerType... I>(IntegerType j, std::integer_sequence<IntegerType, I...>) {  \
                    ((std::forward<TLhsMatrix>(A)(I, j) Operator k), ...);                        \
                };                                                                                \
            auto fCols = [&]<IntegerType... J>(std::integer_sequence<IntegerType, J...>) {        \
                (fRows(J, std::make_integer_sequence<IntegerType, LhsMatrixType::kRows>()), ...); \
            };                                                                                    \
            fCols(std::make_integer_sequence<IntegerType, LhsMatrixType::kCols>());               \
        }                                                                                         \
    }

DefineMatrixMatrixAssign(Assign, =);
DefineMatrixMatrixAssign(AddAssign, +=);
DefineMatrixMatrixAssign(SubtractAssign, -=);
DefineMatrixScalarAssign(AssignScalar, =);
DefineMatrixScalarAssign(AddAssignScalar, +=);
DefineMatrixScalarAssign(SubtractAssignScalar, -=);
DefineMatrixScalarAssign(MultiplyAssign, *=);
DefineMatrixScalarAssign(DivideAssign, /=);

#define PBAT_MINI_ASSIGN_API(SelfType)                     \
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