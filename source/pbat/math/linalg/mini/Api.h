#ifndef PBAT_MATH_LINALG_MINI_API_H
#define PBAT_MATH_LINALG_MINI_API_H

#include "Assign.h"
#include "Concepts.h"
#include "SubMatrix.h"
#include "Transpose.h"

#define PBAT_MINI_READ_API(SelfType)        \
    PBAT_MINI_DIMENSIONS_API                \
    PBAT_MINI_CONST_SUBMATRIX_API(SelfType) \
    PBAT_MINI_CONST_TRANSPOSE_API(SelfType)

#define PBAT_MINI_READ_WRITE_API(SelfType)  \
    PBAT_MINI_DIMENSIONS_API                \
    PBAT_MINI_ASSIGN_API(SelfType)          \
    PBAT_MINI_SUBMATRIX_API(SelfType)       \
    PBAT_MINI_CONST_SUBMATRIX_API(SelfType) \
    PBAT_MINI_TRANSPOSE_API(SelfType)       \
    PBAT_MINI_CONST_TRANSPOSE_API(SelfType)

#endif // PBAT_MATH_LINALG_MINI_API_H