#include "Primitives.cuh"
#include "pbat/profiling/Profiling.h"

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
    PBAT_PROFILE_NAMED_SCOPE("gpu.geometry.Points.Construct");

    auto const Vgpu = V.cast<GpuScalar>().eval();
    std::array<thrust::device_vector<GpuScalar>*, 3> dcoords{&x, &y, &z};
    for (auto d = 0; d < 3; ++d)
    {
        thrust::copy(Vgpu.row(d).begin(), Vgpu.row(d).end(), dcoords[d]->begin());
    }
}

Simplices::Simplices(Eigen::Ref<IndexMatrixX const> const& C) : eSimplexType(), inds()
{
    PBAT_PROFILE_NAMED_SCOPE("gpu.geometry.Simplices.Construct");

    if ((C.rows() < static_cast<int>(ESimplexType::Vertex)) or
        (C.rows() > static_cast<int>(ESimplexType::Tetrahedron)))
    {
        std::string const what =
            "Expected cell index array with either 1,2,3 or 4 rows, corresponding to "
            "vertex,edge,triangle "
            "or tetrahedron simplices, but got " +
            std::to_string(C.rows()) + " rows instead ";
        throw std::invalid_argument(what);
    }

    eSimplexType = static_cast<ESimplexType>(C.rows());
    inds.resize(C.size());
    auto const Cgpu = C.cast<GpuIndex>().eval();
    thrust::copy(Cgpu.data(), Cgpu.data() + Cgpu.size(), inds.begin());
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>
#include <vector>

TEST_CASE("[gpu][geometry] Simplices")
{
    using namespace pbat;
    MatrixX V(3, 4);
    // clang-format off
    V << 0., 1., 2., 3.,
         0., 0., 0., 0.,
         0., 10., 20., 30.;
    // clang-format on
    IndexMatrixX E(2, 3);
    // clang-format off
    E << 1, 0, 2,
         2, 1, 3;
    // clang-format on
    gpu::geometry::Points P(V);
    std::vector<GpuScalar> const PxGpu{P.x.begin(), P.x.end()};
    std::vector<GpuScalar> const PyGpu{P.y.begin(), P.y.end()};
    std::vector<GpuScalar> const PzGpu{P.z.begin(), P.z.end()};
    std::vector<GpuScalar> const PxEigen{V.row(0).begin(), V.row(0).end()};
    std::vector<GpuScalar> const PyEigen{V.row(1).begin(), V.row(1).end()};
    std::vector<GpuScalar> const PzEigen{V.row(2).begin(), V.row(2).end()};
    CHECK_EQ(PxGpu, PxEigen);
    CHECK_EQ(PyGpu, PyEigen);
    CHECK_EQ(PzGpu, PzEigen);

    gpu::geometry::Simplices S(E);
    CHECK_EQ(S.eSimplexType, gpu::geometry::Simplices::ESimplexType::Edge);
    std::vector<GpuIndex> indsGpu{S.inds.begin(), S.inds.end()};
    auto const Egpu = E.cast<GpuIndex>().eval();
    std::vector<GpuIndex> indsEigen{Egpu.data(), Egpu.data() + Egpu.size()};
    CHECK_EQ(indsGpu, indsEigen);
}