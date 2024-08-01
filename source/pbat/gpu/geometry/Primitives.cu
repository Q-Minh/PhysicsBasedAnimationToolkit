#include "Primitives.cuh"

#include <array>
#include <exception>
#include <string>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace pbat {
namespace gpu {
namespace geometry {

Points::Points(Eigen::Ref<MatrixX const> const& V) : x()
{
    for (auto d = 0; d < 3; ++d)
        x[d].resize(V.cols());
    auto const Vgpu = V.cast<GpuScalar>().eval();
    for (auto d = 0; d < 3; ++d)
    {
        thrust::copy(Vgpu.row(d).begin(), Vgpu.row(d).end(), x[d].begin());
    }
}

std::array<GpuScalar const*, 3> Points::Raw() const
{
    return {
        thrust::raw_pointer_cast(x[0].data()),
        thrust::raw_pointer_cast(x[1].data()),
        thrust::raw_pointer_cast(x[2].data())};
}

std::array<GpuScalar*, 3> Points::Raw()
{
    return {
        thrust::raw_pointer_cast(x[0].data()),
        thrust::raw_pointer_cast(x[1].data()),
        thrust::raw_pointer_cast(x[2].data())};
}

Simplices::Simplices(Eigen::Ref<IndexMatrixX const> const& C) : eSimplexType(), inds()
{
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
    for (auto m = 0; m < 4; ++m)
        inds[m].resize(C.cols());
    auto Cgpu = C.cast<GpuIndex>().eval();
    for (auto m = 0; m < C.rows(); ++m)
    {
        thrust::copy(Cgpu.row(m).begin(), Cgpu.row(m).end(), inds[m].begin());
    }
    auto const ninds = (-Cgpu.row(0).array() - 1).eval();
    for (auto m = C.rows(); m < 4; ++m)
    {
        thrust::copy(ninds.data(), ninds.data() + ninds.size(), inds[m].begin());
    }
}

GpuIndex Simplices::NumberOfSimplices() const
{
    return static_cast<GpuIndex>(inds[0].size());
}

std::array<GpuIndex const*, 4> Simplices::Raw() const
{
    return {
        thrust::raw_pointer_cast(inds[0].data()),
        thrust::raw_pointer_cast(inds[1].data()),
        thrust::raw_pointer_cast(inds[2].data()),
        thrust::raw_pointer_cast(inds[3].data())};
}

std::array<GpuIndex*, 4> Simplices::Raw()
{
    return {
        thrust::raw_pointer_cast(inds[0].data()),
        thrust::raw_pointer_cast(inds[1].data()),
        thrust::raw_pointer_cast(inds[2].data()),
        thrust::raw_pointer_cast(inds[3].data())};
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
    for (auto d = 0; d < P.x.size(); ++d)
    {
        std::vector<GpuScalar> const PxGpu{P.x[d].begin(), P.x[d].end()};
        std::vector<GpuScalar> const PxEigen{V.row(d).begin(), V.row(d).end()};
        CHECK_EQ(PxGpu, PxEigen);
    }
    gpu::geometry::Simplices S(E);
    CHECK_EQ(S.eSimplexType, gpu::geometry::Simplices::ESimplexType::Edge);
    auto const nSimplexVertices = static_cast<int>(S.eSimplexType);
    auto const Egpu             = E.cast<GpuIndex>().eval();
    for (auto m = 0; m < nSimplexVertices; ++m)
    {
        std::vector<GpuIndex> indsGpu{S.inds[m].begin(), S.inds[m].end()};
        std::vector<GpuIndex> indsEigen{Egpu.row(m).begin(), Egpu.row(m).end()};
        CHECK_EQ(indsGpu, indsEigen);
    }
    auto const nindsEigen = (-Egpu.row(0).array() - 1).eval();
    for (auto m = nSimplexVertices; m < S.inds.size(); ++m)
    {
        std::vector<GpuIndex> indsGpu{S.inds[m].begin(), S.inds[m].end()};
        std::vector<GpuIndex> indsEigen{nindsEigen.begin(), nindsEigen.end()};
        CHECK_EQ(indsGpu, indsEigen);
    }
}