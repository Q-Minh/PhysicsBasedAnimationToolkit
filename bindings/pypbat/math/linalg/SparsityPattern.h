#ifndef PBAT_MATH_LINALG_SPARSITY_PATTERN_H
#define PBAT_MATH_LINALG_SPARSITY_PATTERN_H

#include <pbat/math/linalg/SparsityPattern.h>
#include <pybind11/pybind11.h>

namespace pbat::py::math::linalg {

// TODO: Write bindings for SparsityPattern
// struct SparsityPattern
// {
//     Eigen::StorageOptions mOptions;
//     enum class EIndexType { Int32, Int64 } mIndexType;
//     void* mImpl;

//     SparsityPattern(
//         Eigen::StorageOptions options = Eigen::ColMajor,
//         EIndexType indexType          = EIndexType::Int32);

//     template <class Func>
//     void Apply(Func f)
//     {
//         if (mOptions == Eigen::ColMajor)
//         {
//             switch (mIndexType)
//             {
//                 case EIndexType::Int32: {
//                     using SparsityPatternType =
//                         pbat::math::linalg::SparsityPattern<std::int32_t, Eigen::ColMajor>;
//                     f.template <SparsityPatternType>
//                     (static_cast<SparsityPatternType*>(mImpl));
//                     break;
//                 }
//                 case EIndexType::Int64: {
//                     using SparsityPatternType =
//                         pbat::math::linalg::SparsityPattern<std::int64_t, Eigen::ColMajor>;
//                     f.template <SparsityPatternType>
//                     (static_cast<SparsityPatternType*>(mImpl));
//                     break;
//                 }
//             }
//         }
//         else
//         {
//             switch (mIndexType)
//             {
//                 case EIndexType::Int32: {
//                     using SparsityPatternType =
//                         pbat::math::linalg::SparsityPattern<std::int32_t, Eigen::RowMajor>;
//                     f.template <SparsityPatternType>
//                     (static_cast<SparsityPatternType*>(mImpl));
//                     break;
//                 }
//                 case EIndexType::Int64: {
//                     using SparsityPatternType =
//                         pbat::math::linalg::SparsityPattern<std::int64_t, Eigen::RowMajor>;
//                     f.template <SparsityPatternType>
//                     (static_cast<SparsityPatternType*>(mImpl));
//                     break;
//                 }
//             }
//         }
//     }

//     ~SparsityPattern();
// };

void BindSparsityPattern(pybind11::module& m);

} // namespace pbat::py::math::linalg

#endif // PBAT_MATH_LINALG_SPARSITY_PATTERN_H