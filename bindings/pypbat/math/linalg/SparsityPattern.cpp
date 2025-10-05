#include "SparsityPattern.h"

#include <pbat/common/ConstexprFor.h>
#include <nanobind/eigen/dense.h>

namespace pbat::py::math::linalg {

// TODO: Write bindings for SparsityPattern

// SparsityPattern::SparsityPattern(Eigen::StorageOptions options, EIndexType indexType)
//     : mOptions(options), mIndexType(indexType), mImpl(nullptr)
// {
//     if (mOptions == Eigen::ColMajor)
//     {
//         switch (mIndexType)
//         {
//             case EIndexType::Int32:
//                 mImpl = new pbat::math::linalg::SparsityPattern<std::int32_t, Eigen::ColMajor>();
//                 break;
//             case EIndexType::Int64:
//                 mImpl = new pbat::math::linalg::SparsityPattern<std::int64_t, Eigen::ColMajor>();
//                 break;
//         }
//     }
//     else
//     {
//         switch (mIndexType)
//         {
//             case EIndexType::Int32:
//                 mImpl = new pbat::math::linalg::SparsityPattern<std::int32_t, Eigen::RowMajor>();
//                 break;
//             case EIndexType::Int64:
//                 mImpl = new pbat::math::linalg::SparsityPattern<std::int64_t, Eigen::RowMajor>();
//                 break;
//         }
//     }
// }

// SparsityPattern::~SparsityPattern()
// {
//     if (mImpl == nullptr)
//         return;

//     if (mOptions == Eigen::ColMajor)
//     {
//         switch (mIndexType)
//         {
//             case EIndexType::Int32:
//                 delete static_cast<
//                     pbat::math::linalg::SparsityPattern<std::int32_t, Eigen::ColMajor>*>(mImpl);
//                 break;
//             case EIndexType::Int64:
//                 delete static_cast<
//                     pbat::math::linalg::SparsityPattern<std::int64_t, Eigen::ColMajor>*>(mImpl);
//                 break;
//         }
//     }
//     else
//     {
//         switch (mIndexType)
//         {
//             case EIndexType::Int32:
//                 delete static_cast<
//                     pbat::math::linalg::SparsityPattern<std::int32_t, Eigen::RowMajor>*>(mImpl);
//                 break;
//             case EIndexType::Int64:
//                 delete static_cast<
//                     pbat::math::linalg::SparsityPattern<std::int64_t, Eigen::RowMajor>*>(mImpl);
//                 break;
//         }
//     }
// }

void BindSparsityPattern([[maybe_unused]] nanobind::module_& m)
{
    namespace nb = nanobind;
}

} // namespace pbat::py::math::linalg