// clang-format off
#include "pbat/gpu/DisableWarnings.h"
// clang-format on

#include "PrimitivesImpl.cuh"

#include <array>
#include <exception>
#include <string>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace pbat {
namespace gpu {
namespace geometry {

PointsImpl::PointsImpl(Eigen::Ref<GpuMatrixX const> const& V) : x()
{
    Update(V);
}

std::size_t PointsImpl::NumberOfPoints() const
{
    return x.Size();
}

std::size_t PointsImpl::Dimensions() const
{
    return x.Dimensions();
}

void PointsImpl::Update(Eigen::Ref<GpuMatrixX const> const& V)
{
    for (auto d = 0; d < 3; ++d)
        x[d].resize(V.cols());
    for (auto d = 0; d < 3; ++d)
    {
        thrust::copy(V.row(d).begin(), V.row(d).end(), x[d].begin());
    }
}

SimplicesImpl::SimplicesImpl(Eigen::Ref<GpuIndexMatrixX const> const& C) : eSimplexType(), inds()
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
    for (auto m = 0; m < C.rows(); ++m)
    {
        thrust::copy(C.row(m).begin(), C.row(m).end(), inds[m].begin());
    }
    auto const ninds = (-C.row(0).array() - 1).eval();
    for (auto m = C.rows(); m < 4; ++m)
    {
        thrust::copy(ninds.begin(), ninds.end(), inds[m].begin());
    }
}

GpuIndex SimplicesImpl::NumberOfSimplices() const
{
    return static_cast<GpuIndex>(inds.Size());
}

BodiesImpl::BodiesImpl(Eigen::Ref<GpuIndexVectorX const> const& B)
    : body(B.size()), nBodies(static_cast<GpuIndex>(B.maxCoeff() + 1))
{
    thrust::copy(B.data(), B.data() + B.size(), body.Data());
}

GpuIndex BodiesImpl::NumberOfBodies() const
{
    return nBodies;
}

} // namespace geometry
} // namespace gpu
} // namespace pbat

#include <doctest/doctest.h>
#include <vector>

TEST_CASE("[gpu][geometry] Simplices")
{
    using namespace pbat;
    GpuMatrixX V(3, 4);
    // clang-format off
    V << 0., 1., 2., 3.,
         0., 0., 0., 0.,
         0., 10., 20., 30.;
    // clang-format on
    GpuIndexMatrixX E(2, 3);
    // clang-format off
    E << 1, 0, 2,
         2, 1, 3;
    // clang-format on
    gpu::geometry::PointsImpl P(V);
    for (auto d = 0; d < P.x.Dimensions(); ++d)
    {
        std::vector<GpuScalar> const PxGpu{P.x[d].begin(), P.x[d].end()};
        std::vector<GpuScalar> const PxEigen{V.row(d).begin(), V.row(d).end()};
        CHECK_EQ(PxGpu, PxEigen);
    }
    gpu::geometry::SimplicesImpl S(E);
    CHECK_EQ(S.eSimplexType, gpu::geometry::SimplicesImpl::ESimplexType::Edge);
    auto const nSimplexVertices = static_cast<int>(S.eSimplexType);
    for (auto m = 0; m < nSimplexVertices; ++m)
    {
        std::vector<GpuIndex> indsGpu{S.inds[m].begin(), S.inds[m].end()};
        std::vector<GpuIndex> indsEigen{E.row(m).begin(), E.row(m).end()};
        CHECK_EQ(indsGpu, indsEigen);
    }
    auto const nindsEigen = (-E.row(0).array() - 1).eval();
    for (auto m = nSimplexVertices; m < S.inds.Dimensions(); ++m)
    {
        std::vector<GpuIndex> indsGpu{S.inds[m].begin(), S.inds[m].end()};
        std::vector<GpuIndex> indsEigen{nindsEigen.begin(), nindsEigen.end()};
        CHECK_EQ(indsGpu, indsEigen);
    }
}