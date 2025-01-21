#ifndef PBAT_GPU_GEOMETRY_IMPL_BVHQUERYKERNELS_H
#define PBAT_GPU_GEOMETRY_IMPL_BVHQUERYKERNELS_H

#include "BvhQuery.cuh"
#include "Primitives.cuh"
#include "pbat/HostDevice.h"
#include "pbat/common/Queue.h"
#include "pbat/common/Stack.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/geometry/Morton.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/gpu/Aliases.h"
#include "pbat/gpu/common/SynchronizedList.cuh"
#include "pbat/math/linalg/mini/Mini.h"

#include <array>

namespace pbat {
namespace gpu {
namespace geometry {
namespace impl {
namespace BvhQueryImplKernels {

namespace mini = pbat::math::linalg::mini;

struct FComputeAabb
{
    PBAT_DEVICE void operator()(int s)
    {
        for (auto d = 0; d < 3; ++d)
        {
            b[d][s] = x[d][inds[0][s]];
            e[d][s] = x[d][inds[0][s]];
            for (auto m = 1; m < nSimplexVertices; ++m)
            {
                b[d][s] = min(b[d][s], x[d][inds[m][s]]);
                e[d][s] = max(e[d][s], x[d][inds[m][s]]);
            }
            b[d][s] -= r;
            e[d][s] += r;
        }
    }

    std::array<GpuScalar const*, 3> x;
    std::array<GpuIndex const*, 4> inds;
    int nSimplexVertices;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    GpuScalar r;
};

struct FComputeMortonCode
{
    using MortonCodeType = pbat::geometry::MortonCodeType;

    PBAT_DEVICE void operator()(int s)
    {
        // Compute Morton code of the centroid of the bounding box of simplex s
        std::array<GpuScalar, 3> c{};
        for (auto d = 0; d < 3; ++d)
        {
            auto cd = GpuScalar{0.5} * (b[d][s] + e[d][s]);
            c[d]    = (cd - sb[d]) / sbe[d];
        }
        using pbat::geometry::Morton3D;
        morton[s] = Morton3D(c);
    }

    std::array<GpuScalar, 3> sb;
    std::array<GpuScalar, 3> sbe;
    std::array<GpuScalar*, 3> b;
    std::array<GpuScalar*, 3> e;
    MortonCodeType* morton;
};

struct FDetectOverlaps
{
    using OverlapType = typename BvhQuery::OverlapType;

    PBAT_DEVICE bool AreSimplicesTopologicallyAdjacent(GpuIndex si, GpuIndex sj) const
    {
        auto count{0};
        for (auto i = 0; i < queryInds.size(); ++i)
            for (auto j = 0; j < inds.size(); ++j)
                count += (queryInds[i][si] == inds[j][sj]);
        return count > 0;
    }

    PBAT_DEVICE bool AreBoxesOverlapping(GpuIndex i, GpuIndex j) const
    {
        // clang-format off
        return (queryE[0][i] >= b[0][j]) and (queryB[0][i] <= e[0][j]) and
               (queryE[1][i] >= b[1][j]) and (queryB[1][i] <= e[1][j]) and
               (queryE[2][i] >= b[2][j]) and (queryB[2][i] <= e[2][j]);
        // clang-format on
    }

    PBAT_DEVICE bool VertexTetrahedronOverlap(GpuIndex v, GpuIndex t) const
    {
        using namespace mini;
        SVector<GpuScalar, 3> P = FromBuffers<3, 1>(x, queryInds[0][v]);
        SVector<GpuScalar, 3> A = FromBuffers<3, 1>(x, inds[0][t]);
        SVector<GpuScalar, 3> B = FromBuffers<3, 1>(x, inds[1][t]);
        SVector<GpuScalar, 3> C = FromBuffers<3, 1>(x, inds[2][t]);
        SVector<GpuScalar, 3> D = FromBuffers<3, 1>(x, inds[3][t]);
        using pbat::geometry::OverlapQueries::PointTetrahedron3D;
        return PointTetrahedron3D(P, A, B, C, D);
    }

    PBAT_DEVICE bool AreSimplicesOverlapping(GpuIndex si, GpuIndex sj) const
    {
        if (querySimplexType == Simplices::ESimplexType::Vertex and
            targetSimplexType == Simplices::ESimplexType::Tetrahedron)
        {
            return VertexTetrahedronOverlap(si, sj);
        }
        return true;
    }

    PBAT_DEVICE void operator()(auto query)
    {
        // Traverse nodes depth-first starting from the root.
        using pbat::common::Stack;
        Stack<GpuIndex, 64> stack{};
        stack.Push(0);
        do
        {
            assert(not stack.IsFull());
            GpuIndex const node = stack.Pop();
            // Check each child node for overlap.
            GpuIndex const lc            = child[0][node];
            GpuIndex const rc            = child[1][node];
            bool const bLeftBoxOverlaps  = AreBoxesOverlapping(query, lc);
            bool const bRightBoxOverlaps = AreBoxesOverlapping(query, rc);

            // Query overlaps another leaf node -> report collision if topologically separate
            // simplices
            bool const bIsLeftLeaf = lc >= leafBegin;
            if (bLeftBoxOverlaps and bIsLeftLeaf)
            {
                GpuIndex const si = querySimplex[query];
                GpuIndex const sj = simplex[lc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(si, sj) and
                    AreSimplicesOverlapping(si, sj) and not overlaps.Append({si, sj}))
                    break;
            }
            bool const bIsRightLeaf = rc >= leafBegin;
            if (bRightBoxOverlaps and bIsRightLeaf)
            {
                GpuIndex const si = querySimplex[query];
                GpuIndex const sj = simplex[rc - leafBegin];
                if (not AreSimplicesTopologicallyAdjacent(si, sj) and
                    AreSimplicesOverlapping(si, sj) and not overlaps.Append({si, sj}))
                    break;
            }

            // Query overlaps an internal node -> traverse.
            bool const bTraverseLeft  = bLeftBoxOverlaps and not bIsLeftLeaf;
            bool const bTraverseRight = bRightBoxOverlaps and not bIsRightLeaf;
            if (bTraverseLeft)
                stack.Push(lc);
            if (bTraverseRight)
                stack.Push(rc);
        } while (not stack.IsEmpty());
    }

    std::array<GpuScalar const*, 3> x;

    Simplices::ESimplexType querySimplexType;
    GpuIndex* querySimplex;
    std::array<GpuIndex const*, 4> queryInds;
    std::array<GpuScalar*, 3> queryB;
    std::array<GpuScalar*, 3> queryE;

    Simplices::ESimplexType targetSimplexType;
    GpuIndex const* simplex;
    std::array<GpuIndex const*, 4> inds;
    std::array<GpuScalar const*, 3> b;
    std::array<GpuScalar const*, 3> e;
    std::array<GpuIndex const*, 2> child;
    GpuIndex leafBegin;

    common::DeviceSynchronizedList<OverlapType> overlaps;
};

struct FContactPairs
{
    using OverlapType              = typename BvhQuery::OverlapType;
    using NearestNeighbourPairType = typename BvhQuery::NearestNeighbourPairType;

    PBAT_DEVICE GpuScalar MinDistance(
        mini::SVector<GpuScalar, 3> const& X,
        mini::SVector<GpuScalar, 3> const& L,
        mini::SVector<GpuScalar, 3> const& U) const
    {
        using namespace mini;
        SVector<GpuScalar, 3> const DX = Min(U, Max(L, X)) - X;
        return SquaredNorm(DX);
    }
    PBAT_DEVICE GpuScalar MinMaxDistance(
        mini::SVector<GpuScalar, 3> const& X,
        mini::SVector<GpuScalar, 3> const& L,
        mini::SVector<GpuScalar, 3> const& U) const
    {
        using namespace mini;
        SVector<GpuScalar, 3> DXL = Squared(L - X);
        SVector<GpuScalar, 3> DXU = Squared(U - X);
        SVector<GpuScalar, 3> rm  = Min(DXL, DXU);
        SVector<GpuScalar, 3> rM  = Max(DXL, DXU);
        SVector<GpuScalar, 3> d{
            rm(0) + rM(1) + rM(2),
            rM(0) + rm(1) + rM(2),
            rM(0) + rM(1) + rm(2),
        };
        return Min(d);
    }
    PBAT_DEVICE GpuScalar Distance(mini::SVector<GpuScalar, 3> const& P, GpuIndex s) const
    {
        using namespace mini;
        auto A = FromBuffers<3, 1>(x, targetInds[0][s]);
        auto B = FromBuffers<3, 1>(x, targetInds[1][s]);
        auto C = FromBuffers<3, 1>(x, targetInds[2][s]);
        using pbat::geometry::DistanceQueries::PointTriangle;
        return PointTriangle(P, A, B, C);
    }

    struct BoxOrSimplex
    {
        GpuIndex node; ///< BVH node index
        GpuScalar d;   ///< Distance between query and node
    };

    struct BranchAndBound
    {
        using StackType = pbat::common::Stack<BoxOrSimplex, 64>;
        using QueueType = pbat::common::Queue<GpuIndex, 8>;

        PBAT_DEVICE
        BranchAndBound(
            mini::SVector<GpuScalar, 3> const& X,
            GpuScalar R,
            GpuScalar dzero,
            GpuIndex v)
            : stack{}, nearest{}, X(X), R(R), dzero(dzero), v(v)
        {
        }

        StackType stack;
        QueueType nearest;
        mini::SVector<GpuScalar, 3> X;
        GpuScalar R;
        GpuScalar dzero;
        GpuIndex v;
    };

    PBAT_DEVICE void Push(BranchAndBound& traversal, GpuIndex node, GpuScalar dbox) const
    {
        if (node >= leafBegin)
        {
            GpuIndex const s                     = simplex[node - leafBegin];
            bool const bFromDifferentBodies      = body[traversal.v] != body[targetInds[0][s]];
            bool const bAreTopologicallySeparate = (traversal.v != targetInds[0][s]) and
                                                   (traversal.v != targetInds[1][s]) and
                                                   (traversal.v != targetInds[2][s]);
            bool const bIsValidContactPair = bFromDifferentBodies and bAreTopologicallySeparate;
            if (not bIsValidContactPair)
                return;

            GpuScalar d = Distance(traversal.X, s);
            if (d < traversal.R)
            {
                traversal.nearest.Clear();
                traversal.nearest.Push(s);
                traversal.R = d;
            }
            else if (d - traversal.R <= dzero and not traversal.nearest.IsFull())
            {
                traversal.nearest.Push(s);
            }
        }
        else
        {
            traversal.stack.Push({node, dbox});
        }
    }

    PBAT_DEVICE BoxOrSimplex Pop(BranchAndBound& traversal) const
    {
        BoxOrSimplex bos = traversal.stack.Top();
        traversal.stack.Pop();
        return bos;
    }

    PBAT_DEVICE void operator()(OverlapType const& o)
    {
        using namespace mini;
        // Branch and bound over BVH
        GpuIndex const sv = o.first;
        GpuIndex const v  = queryInds[0][sv];
        BranchAndBound traversal{FromBuffers<3, 1>(x, v), R, dzero, v};
        Push(
            traversal,
            0 /* root node */,
            MinDistance(
                traversal.X,
                FromBuffers<3, 1>(b, 0),
                FromBuffers<3, 1>(e, 0)) /* distance from query point to root's aabb */);
        do
        {
            assert(not traversal.stack.IsFull());
            BoxOrSimplex const bos = Pop(traversal);
            if (bos.d > traversal.R)
                continue;

            GpuIndex const lc = child[0][bos.node];
            GpuIndex const rc = child[1][bos.node];

            mini::SVector<GpuScalar, 3> const LL = FromBuffers<3, 1>(b, lc);
            mini::SVector<GpuScalar, 3> const LU = FromBuffers<3, 1>(e, lc);
            mini::SVector<GpuScalar, 3> const RL = FromBuffers<3, 1>(b, rc);
            mini::SVector<GpuScalar, 3> const RU = FromBuffers<3, 1>(e, rc);

            GpuScalar Ldmin    = MinDistance(traversal.X, LL, LU);
            GpuScalar Rdmin    = MinDistance(traversal.X, RL, RU);
            GpuScalar Ldminmax = MinMaxDistance(traversal.X, LL, LU);
            GpuScalar Rdminmax = MinMaxDistance(traversal.X, RL, RU);

            if (Ldmin <= Rdminmax)
                Push(traversal, lc, Ldmin);
            if (Rdmin <= Ldminmax)
                Push(traversal, rc, Rdmin);
        } while (not traversal.stack.IsEmpty());
        // Collect results
        while (not traversal.nearest.IsEmpty())
        {
            GpuIndex s = traversal.nearest.Top();
            traversal.nearest.Pop();
            neighbours.Append({sv, s});
        }
    }

    std::array<GpuScalar const*, 3> x;

    GpuIndex const* body;                     ///< Body indices of points
    std::array<GpuIndex const*, 4> queryInds; ///< Vertex indices of query simplices
    GpuScalar const R;                        ///< Nearest neighbour query search radius
    GpuScalar const dzero; ///< Error tolerance for different distances to be considered the same,
                           ///< i.e. if
                           ///< the distances di,dj to the nearest neighbours i,j are similar, we
                           ///< considered both i and j to be nearest neighbours.

    std::array<GpuIndex const*, 4> targetInds; ///< Vertex indices of target simplices
    GpuIndex const* simplex;                   ///< Target simplices
    std::array<GpuScalar const*, 3> b;         ///< Box beginnings of BVH
    std::array<GpuScalar const*, 3> e;         ///< Box endings of BVH
    std::array<GpuIndex const*, 2> child;      ///< BVH children
    GpuIndex const leafBegin;                  ///< Index to beginning of BVH's leaves array

    common::DeviceSynchronizedList<NearestNeighbourPairType>
        neighbours; ///< Nearest neighbour pairs found
};

} // namespace BvhQueryImplKernels
} // namespace impl
} // namespace geometry
} // namespace gpu
} // namespace pbat

#endif // PBAT_GPU_GEOMETRY_IMPL_BVHQUERYKERNELS_H
