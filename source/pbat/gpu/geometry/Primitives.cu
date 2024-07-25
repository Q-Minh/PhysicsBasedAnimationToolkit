#include "Primitives.cuh"
#include "pbat/common/Hash.h"

#include <array>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <tuple>
#include <unordered_set>

namespace pbat {
namespace gpu {
namespace geometry {

Points::Points(Eigen::Ref<MatrixX const> const& V) : x(V.cols()), y(V.cols()), z(V.cols())
{
    std::array<thrust::device_vector<ScalarType>*, 3> dcoords{&x, &y, &z};
    for (auto d = 0; d < 3; ++d)
    {
        thrust::copy(V.row(d).begin(), V.row(d).end(), dcoords[d]->begin());
    }
}

Simplices::Simplices(Eigen::Ref<IndexMatrixX const> const& C, int simplexTypes)
    : eSimplexTypes(simplexTypes), c(), inds()
{
    using EdgeType        = std::tuple<IndexType, IndexType>;
    using TriangleType    = std::tuple<IndexType, IndexType, IndexType>;
    using TetrahedronType = std::tuple<IndexType, IndexType, IndexType, IndexType>;

    struct EdgeHash
    {
        std::size_t operator()(EdgeType const& simplex) const
        {
            return common::HashCombine(std::get<0>(simplex), std::get<1>(simplex));
        }
    };
    struct TriangleHash
    {
        std::size_t operator()(TriangleType const& simplex) const
        {
            return common::HashCombine(
                std::get<0>(simplex),
                std::get<1>(simplex),
                std::get<2>(simplex));
        }
    };
    struct TetrahedronHash
    {
        std::size_t operator()(TetrahedronType const& simplex) const
        {
            return common::HashCombine(
                std::get<0>(simplex),
                std::get<1>(simplex),
                std::get<2>(simplex),
                std::get<3>(simplex));
        }
    };

    bool const bIsEdgeMesh        = C.rows() == 2;
    bool const bIsTriangleMesh    = C.rows() == 3;
    bool const bIsTetrahedralMesh = C.rows() == 4;
    std::unordered_set<IndexType> V{};
    if (eSimplexTypes & ESimplexType::Vertex)
    {
        V.reserve(C.cols() * C.rows());
    }
    std::unordered_set<IndexType> E{};
    if (eSimplexTypes & ESimplexType::Edge)
    {
        E.reserve(
            bIsEdgeMesh        ? C.cols() :
            bIsTriangleMesh    ? C.cols() * 3 :
            bIsTetrahedralMesh ? C.cols() * 6 :
                                 0);
    }
    std::unordered_set<IndexType> F{};
    if (eSimplexTypes & ESimplexType::Triangle)
    {
        F.reserve(bIsTriangleMesh ? C.cols() : bIsTetrahedralMesh ? C.cols() * 4 : 0);
    }
    std::unordered_set<IndexType> T{};
    if (eSimplexTypes & ESimplexType::Tetrahedron)
    {
        T.reserve(bIsTetrahedralMesh ? C.cols() : 0);
    }
}

} // namespace geometry
} // namespace gpu
} // namespace pbat