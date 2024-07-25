#include "Primitives.cuh"

#include <array>
#include <exception>
#include <string>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace pbat {
namespace gpu {
namespace geometry {

Points::Points(Eigen::Ref<MatrixX const> const& V) : x(V.cols()), y(V.cols()), z(V.cols())
{
    auto const Vgpu = V.cast<GpuScalar>().eval();
    std::array<thrust::device_vector<GpuScalar>*, 3> dcoords{&x, &y, &z};
    for (auto d = 0; d < 3; ++d)
    {
        thrust::copy(Vgpu.row(d).begin(), Vgpu.row(d).end(), dcoords[d]->begin());
    }
}

Simplices::Simplices(Eigen::Ref<IndexMatrixX const> const& C) : eSimplexType(), inds()
{
    if ((C.rows() < static_cast<int>(ESimplexType::Edge)) or
        (C.rows() > static_cast<int>(ESimplexType::Tetrahedron)))
    {
        std::string const what =
            "Expected cell index array with either 2,3 or 4 rows, corresponding to edge,triangle "
            "or tetrahedron simplices, but got " +
            std::to_string(C.rows()) + " rows instead ";
        throw std::invalid_argument(what);
    }

    eSimplexType = static_cast<ESimplexType>(C.rows());
    inds.resize(C.size());
    auto const Cgpu = C.cast<GpuIndex>().eval();
    thrust::copy(thrust::device, Cgpu.data(), Cgpu.data() + Cgpu.size(), inds.begin());
}

} // namespace geometry
} // namespace gpu
} // namespace pbat