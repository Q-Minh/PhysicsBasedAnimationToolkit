/**
 * @file Ogc.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Offset Geometry Contact (OGC) algorithm API.
 * @version 0.1
 * @date 2025-12-10
 * @copyright Copyright (c) 2025
 */
#ifndef PBAT_SIM_CONTACT_OGC_OGC_H
#define PBAT_SIM_CONTACT_OGC_OGC_H

#include "Enums.h"
#include "Input.h"
#include "Params.h"
#include "State.h"
#include "pbat/common/Atomic.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <tbb/parallel_for.h>

namespace pbat::sim::contact::ogc {

/**
 * @brief Performs vertex-facet contact detection.
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param input OGC's input
 * @param params OGC's parameters
 * @param state OGC's state
 * @pre state.PrepareForExecution() has been called
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void VertexFacetContactDetection(
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params,
    State<TScalar, TIndex>& state);

/**
 * @brief Performs edge-edge contact detection.
 *
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param input OGC's input
 * @param params OGC's parameters
 * @param state OGC's state
 * @pre state.PrepareForExecution() has been called
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void EdgeEdgeContactDetection(
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params,
    State<TScalar, TIndex>& state);

/**
 * @brief Updates displacement bounds.
 *
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param input OGC's input
 * @param params OGC's parameters
 * @param state OGC's state
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void UpdateDisplacementBounds(
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params,
    State<TScalar, TIndex>& state);

/**
 * @brief Computes the triangle face (vertex, edge or triangle) nearest to a point's projection on
 * a triangle.
 * @param u First barycentric coordinate of the closest point on the triangle
 * @param v Second barycentric coordinate of the closest point on the triangle
 * @param w Third barycentric coordinate of the closest point on the triangle
 * @return The pair (a, eFace), where a is either a local vertex index or edge index, and eFace
 * indicates the type of face, i.e. (0 | 1 | 2) -> (triangle | edge | vertex)
 */
template <common::CFloatingPoint TScalar>
std::pair<int, int> ClosestFaceFacetToVertex(TScalar u, TScalar v, TScalar w);

/**
 * @brief Computes the edge face (vertex or edge) nearest to the closest points on two edges.
 *
 * @tparam TScalar
 * @param s Barycentric coordinate of closest point on edge 1
 * @param t Barycentric coordinate of closest point on edge 2
 * @param e1 Edge index of edge 1
 * @param e2 Edge index of edge 2
 * @param e1v Vertex indices of edge 1
 * @param e2v Vertex indices of edge 2
 * @return The tuple (a1, eFace1, a2, eFace2), where a1 and a2 are either point indices or
 * edge indices, and eFace1 and eFace2 indicate the type of face (vertex or edge), i.e. (0 | 1) ->
 * (edge | vertex)
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
std::tuple<int, int, int, int> ClosestFaceEdgeToEdge(
    TScalar s,
    TScalar t,
    TIndex e1,
    TIndex e2,
    std::array<TIndex, 2> e1v,
    std::array<TIndex, 2> e2v);

/**
 * @brief Vectorize contact face index based on face type.
 *
 * Given a triangle mesh face `f`, a local index `alocal` (local vertex or local half-edge index),
 * and a face-type tag `eFace` where 0=triangle, 1=edge, 2=vertex, this computes the unified
 * contact index `a` as used by the OGC contact sets.
 *
 * Definition:
 * - Triangle-face:   a = f
 * - Edge-face:       a = 3*f + alocal  (half-edge index within face f)
 * - Vertex-face:     a = F(alocal, f)  (global point index)
 *
 * @tparam TIndex Index type
 * @tparam TDerivedF Derived Eigen type for face connectivity (`3 x |# faces|`)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param f Face index
 * @param alocal Local index within face f (0..2)
 * @param eFace Face type tag: 0=triangle, 1=edge, 2=vertex
 * @return Vectorized contact index `a`
 */
template <common::CIndex TIndex, class TDerivedF>
TIndex
VertexFacetContactFaceIndex(Eigen::DenseBase<TDerivedF> const& F, TIndex f, int alocal, int eFace);

/**
 * @brief Determines if point x is in the vertex feasible region of vertex i.
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 * @tparam TDerivedX Derived type for point positions
 * @tparam TDerivedF Derived type for triangle vertex indices
 * @tparam TDerivedGVHEp Derived type for vertex to half-edge prefix
 * @tparam TDerivedGVHEadj Derived type for vertex to half-edge adjacency
 * @param X `3 x |# points|` point positions
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param GVHEp `|# vertices + 1|` vertex to half-edge prefix
 * @param GVHEadj `|# half edges|` vertex to half-edge adjacency
 * @param x `3 x 1` query point
 * @param i Point (global) index corresponding to vertex
 * @return true if in vertex feasible region; false otherwise
 */
template <
    common::CFloatingPoint TScalar,
    common::CIndex TIndex,
    class TDerivedX,
    class TDerivedF,
    class TDerivedGVHEp,
    class TDerivedGVHEadj>
bool IsVertexFeasible(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedF> const& F,
    Eigen::DenseBase<TDerivedGVHEp> const& GVHEp,
    Eigen::DenseBase<TDerivedGVHEadj> const& GVHEadj,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex i);

/**
 * @brief Determines if point x is in the edge feasible region of half-edge he of face fi.
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 * @tparam TDerivedX Derived type for point positions
 * @tparam TDerivedF Derived type for triangle vertex indices
 * @tparam TDerivedGHEF Derived type for half-edge to face adjacency
 * @param X `3 x |# points|` point positions
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param GHEF `2 x |# half edges|` half-edge to (adjacent face, opposite face)
 * @param x `3 x 1` query point
 * @param fi Face index of half-edge he
 * @param he Half-edge index
 * @return true if in edge feasible region; false otherwise
 */
template <
    common::CFloatingPoint TScalar,
    common::CIndex TIndex,
    class TDerivedX,
    class TDerivedF,
    class TDerivedGHEF>
bool IsEdgeFeasible(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedF> const& F,
    Eigen::DenseBase<TDerivedGHEF> const& GHEF,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex fi,
    TIndex he);

namespace detail {

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct VertexFacetRTCCollideFuncParams
{
    Input<TScalar, TIndex> const* input; ///< OGC input for the body
    Params<TScalar> const* params;       ///< OGC parameters
    State<TScalar, TIndex>* state;       ///< OGC state for the body
};

/**
 * @brief RTC collide function for dynamic vertex - dynamic facet collisions.
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param userPtr User pointer
 * @param collisions Array of RTC collisions
 * @param nCollisions Number of RTC collisions
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void DynamicVertexFacetRTCCollideFunc(
    void* userPtr,
    struct RTCCollision* collisions,
    unsigned int nCollisions)
{
    auto* data = static_cast<VertexFacetRTCCollideFuncParams<TScalar, TIndex>*>(userPtr);
    Input<TScalar, TIndex> const* input                 = data->input;
    Params<TScalar> const* params                       = data->params;
    State<TScalar, TIndex>* state                       = data->state;
    auto const& X                                       = input->X.value();
    auto const& V                                       = input->V.value();
    auto const& F                                       = input->F.value();
    auto const& VP                                      = input->VP.value();
    auto const& FP                                      = input->FP.value();
    auto const& GVHEp                                   = input->GVHEp.value();
    auto const& GVHEadj                                 = input->GVHEadj.value();
    auto const& GHEF                                    = input->GHEF.value();
    TScalar const r                                     = params->r;
    auto& mVertexLocks                                  = state->mVertexLocks;
    auto& mFacetLocks                                   = state->mFacetLocks;
    auto& dminv                                         = state->dminv;
    auto& dminf                                         = state->dminf;
    std::vector<std::vector<ContactFace<TIndex>>>& FOGC = state->mDynamicContactFacesOfVertex;
    std::vector<std::vector<TIndex>>& VOGC              = state->mDynamicContactVerticesOfTriangle;
    // For each potential contact pair (v,f)
    for (unsigned int ci = 0; ci < nCollisions; ++ci)
    {
        // Get vertex-facet pair (iv, f) where ix is the point index of vertex iv
        RTCCollision const& collision        = collisions[ci];
        TIndex const bv                      = static_cast<TIndex>(collision.geomID0);
        TIndex const bf                      = static_cast<TIndex>(collision.geomID1);
        TIndex const iv                      = input->DynamicVertex(bv, collision.primID0);
        TIndex const ix                      = V(iv);
        TIndex const f                       = input->DynamicFacet(bf, collision.primID1);
        Eigen::Vector<TIndex, 3> const finds = F.col(f);
        // Avoid contact with adjacent triangle
        if ((finds.array() == ix).any())
            continue;
        // Compute distance from vertex i to triangle f via closest point projection
        Eigen::Vector<TScalar, 3> const xi    = X.col(ix);
        Eigen::Matrix<TScalar, 3, 3> const xf = X(Eigen::placeholders::all, finds);
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::SVector;
        SVector<TScalar, 3> uvw = geometry::ClosestPointQueries::UvwPointInTriangle(
            FromEigen(xi),
            FromEigen(xf.col(0)),
            FromEigen(xf.col(1)),
            FromEigen(xf.col(2)));
        Eigen::Vector<TScalar, 3> dx2f =
            (xi - (uvw(0) * xf.col(0) + uvw(1) * xf.col(1) + uvw(2) * xf.col(2)));
        TScalar d2 = dx2f.squaredNorm();
        // Update triangle and vertex displacement bounds
        common::AtomicMin(dminv(iv), d2);
        common::AtomicMin(dminf(f), d2);
        // No contact if outside contact radius
        bool const bInContactRadius = (d2 < r * r);
        if (not bInContactRadius)
            continue;
        // Determine face (vertex, edge, triangle) closest to vertex iv
        auto const [alocal, eFace] = ClosestFaceFacetToVertex(uvw(0), uvw(1), uvw(2));
        // Vectorize contact face index a based on its type (triangle | edge | vertex)
        TIndex const a = VertexFacetContactFaceIndex(F, f, alocal, eFace);
        // Synchronize reads/writes to vertex iv's contact sets
        common::AtomicExecute(mVertexLocks(iv), [&]() {
            // Brute-force duplicate search
            bool const bDuplicateContact = std::any_of(
                FOGC[iv].begin(),
                FOGC[iv].end(),
                [&](ContactFace<TIndex> const& contactFace) {
                    return contactFace.a == a and contactFace.eFace == eFace;
                });
            if (bDuplicateContact)
                return;
            // Update contact face sets
            auto const fUpdateContactSets = [&]() {
                FOGC[iv].emplace_back(a, eFace);
                common::AtomicExecute(mFacetLocks(f), [&]() { VOGC[f].emplace_back(ix); });
            };
            switch (static_cast<EVertexFacetClosestFaceType>(eFace))
            {
                case EVertexFacetClosestFaceType::Vertex: {
                    if (IsVertexFeasible(X, F, GVHEp, GVHEadj, xi, a))
                        fUpdateContactSets();
                    break;
                }
                case EVertexFacetClosestFaceType::Edge: {
                    if (IsEdgeFeasible(X, F, GHEF, xi, f, a))
                        fUpdateContactSets();
                    break;
                }
                default /* triangle */: {
                    fUpdateContactSets();
                    break;
                }
            }
        });
    }
}

/**
 * @brief RTC collide function for dynamic vertex - static facet collisions.
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param userPtr User pointer
 * @param collisions Array of RTC collisions
 * @param nCollisions Number of RTC collisions
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void DynamicVertexStaticFacetRTCCollideFunc(
    void* userPtr,
    struct RTCCollision* collisions,
    unsigned int nCollisions)
{
    auto* data = static_cast<VertexFacetRTCCollideFuncParams<TScalar, TIndex>*>(userPtr);
    Input<TScalar, TIndex> const* input                 = data->input;
    Params<TScalar> const* params                       = data->params;
    State<TScalar, TIndex>* state                       = data->state;
    auto const& X                                       = input->X.value();
    auto const& V                                       = input->V.value();
    auto const& VP                                      = input->VP.value();
    auto const& Xenv                                    = input->Venv.value();
    auto const& Fenv                                    = input->Fenv.value();
    auto const& GVHEenvp                                = input->GVHEenvp.value();
    auto const& GVHEenvadj                              = input->GVHEenvadj.value();
    auto const& GHEFenv                                 = input->GHEFenv.value();
    TScalar const r                                     = params->r;
    auto& mVertexLocks                                  = state->mVertexLocks;
    auto& dminv                                         = state->dminv;
    std::vector<std::vector<ContactFace<TIndex>>>& FOGC = state->mStaticContactFacesOfVertex;
    // For each potential contact pair (v,f)
    for (unsigned int ci = 0; ci < nCollisions; ++ci)
    {
        // Get vertex-facet pair (iv, f) where ix is the point index of vertex iv
        RTCCollision const& collision        = collisions[ci];
        TIndex const bv                      = static_cast<TIndex>(collision.geomID0);
        TIndex const iv                      = input->DynamicVertex(bv, collision.primID0);
        TIndex const ix                      = V(iv);
        TIndex const f                       = static_cast<TIndex>(collision.primID1);
        Eigen::Vector<TIndex, 3> const finds = Fenv.col(f);
        // Compute distance from vertex i to triangle f via closest point projection
        Eigen::Vector<TScalar, 3> const xi    = X.col(ix);
        Eigen::Matrix<TScalar, 3, 3> const xf = Xenv(Eigen::placeholders::all, finds);
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::SVector;
        SVector<TScalar, 3> uvw = geometry::ClosestPointQueries::UvwPointInTriangle(
            FromEigen(xi),
            FromEigen(xf.col(0)),
            FromEigen(xf.col(1)),
            FromEigen(xf.col(2)));
        Eigen::Vector<TScalar, 3> dx2f =
            (xi - (uvw(0) * xf.col(0) + uvw(1) * xf.col(1) + uvw(2) * xf.col(2)));
        TScalar d2 = dx2f.squaredNorm();
        // Update triangle and vertex displacement bounds
        common::AtomicMin(dminv(iv), d2);
        // No contact if outside contact radius
        bool const bInContactRadius = (d2 < r * r);
        if (not bInContactRadius)
            continue;
        // Determine face (vertex, edge, triangle) closest to vertex iv
        auto const [alocal, eFace] = ClosestFaceFacetToVertex(uvw(0), uvw(1), uvw(2));
        // Vectorize contact face index a based on its type (triangle | edge | vertex)
        TIndex const a = VertexFacetContactFaceIndex(Fenv, f, alocal, eFace);
        // Synchronize reads/writes to vertex iv's contact sets
        common::AtomicExecute(mVertexLocks(iv), [&]() {
            // Brute-force duplicate search
            bool const bDuplicateContact = std::any_of(
                FOGC[iv].begin(),
                FOGC[iv].end(),
                [&](ContactFace<TIndex> const& contactFace) {
                    return contactFace.a == a and contactFace.eFace == eFace;
                });
            if (bDuplicateContact)
                return;
            // Update contact face sets
            auto const fUpdateContactSets = [&]() {
                FOGC[iv].emplace_back(a, eFace);
            };
            switch (static_cast<EVertexFacetClosestFaceType>(eFace))
            {
                case EVertexFacetClosestFaceType::Vertex: {
                    if (IsVertexFeasible(Xenv, Fenv, GVHEenvp, GVHEenvadj, xi, a))
                        fUpdateContactSets();
                    break;
                }
                case EVertexFacetClosestFaceType::Edge: {
                    if (IsEdgeFeasible(Xenv, Fenv, GHEFenv, xi, f, a))
                        fUpdateContactSets();
                    break;
                }
                default /* triangle */: {
                    fUpdateContactSets();
                    break;
                }
            }
        });
    }
}

/**
 * @brief RTC collide function for static vertex - dynamic facet collisions.
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param userPtr User pointer
 * @param collisions Array of RTC collisions
 * @param nCollisions Number of RTC collisions
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void StaticVertexDynamicFacetRTCCollideFunc(
    void* userPtr,
    struct RTCCollision* collisions,
    unsigned int nCollisions)
{
    auto* data = static_cast<VertexFacetRTCCollideFuncParams<TScalar, TIndex>*>(userPtr);
    Input<TScalar, TIndex> const* input    = data->input;
    Params<TScalar> const* params          = data->params;
    State<TScalar, TIndex>* state          = data->state;
    auto const& Xenv                       = input->Venv.value();
    auto const& X                          = input->X.value();
    auto const& F                          = input->F.value();
    auto const& GVHEp                      = input->GVHEp.value();
    auto const& GVHEadj                    = input->GVHEadj.value();
    auto const& GHEF                       = input->GHEF.value();
    TScalar const r                        = params->r;
    auto& mFacetLocks                      = state->mFacetLocks;
    auto& dminf                            = state->dminf;
    std::vector<std::vector<TIndex>>& VOGC = state->mStaticContactVerticesOfTriangle;
    // For each potential contact pair (v,f)
    for (unsigned int ci = 0; ci < nCollisions; ++ci)
    {
        // Get vertex-facet pair (iv, f) where ix is the point index of vertex iv
        RTCCollision const& collision        = collisions[ci];
        TIndex const ix                      = static_cast<TIndex>(collision.primID0);
        TIndex const bf                      = static_cast<TIndex>(collision.geomID1);
        TIndex const f                       = input->DynamicFacet(bf, collision.primID1);
        Eigen::Vector<TIndex, 3> const finds = F.col(f);
        // Compute distance from vertex i to triangle f via closest point projection
        Eigen::Vector<TScalar, 3> const xi    = Xenv.col(ix);
        Eigen::Matrix<TScalar, 3, 3> const xf = X(Eigen::placeholders::all, finds);
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::SVector;
        SVector<TScalar, 3> uvw = geometry::ClosestPointQueries::UvwPointInTriangle(
            FromEigen(xi),
            FromEigen(xf.col(0)),
            FromEigen(xf.col(1)),
            FromEigen(xf.col(2)));
        Eigen::Vector<TScalar, 3> dx2f =
            (xi - (uvw(0) * xf.col(0) + uvw(1) * xf.col(1) + uvw(2) * xf.col(2)));
        TScalar d2 = dx2f.squaredNorm();
        // Update triangle and vertex displacement bounds
        common::AtomicMin(dminf(f), d2);
        // No contact if outside contact radius
        bool const bInContactRadius = (d2 < r * r);
        if (not bInContactRadius)
            continue;
        // Determine face (vertex, edge, triangle) closest to vertex iv
        auto const [alocal, eFace] = ClosestFaceFacetToVertex(uvw(0), uvw(1), uvw(2));
        // Vectorize contact face index a based on its type (triangle | edge | vertex)
        TIndex const a = VertexFacetContactFaceIndex(F, f, alocal, eFace);
        // Update contact face sets
        auto const fUpdateContactSets = [&]() {
            common::AtomicExecute(mFacetLocks(f), [&]() { VOGC[f].emplace_back(ix); });
        };
        switch (static_cast<EVertexFacetClosestFaceType>(eFace))
        {
            case EVertexFacetClosestFaceType::Vertex: {
                if (IsVertexFeasible(X, F, GVHEp, GVHEadj, xi, a))
                    fUpdateContactSets();
                break;
            }
            case EVertexFacetClosestFaceType::Edge: {
                if (IsEdgeFeasible(X, F, GHEF, xi, f, a))
                    fUpdateContactSets();
                break;
            }
            default /* triangle */: {
                fUpdateContactSets();
                break;
            }
        }
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct EdgeEdgeRTCCollideFuncParams
{
    Input<TScalar, TIndex> const* input; ///< OGC input for the body
    Params<TScalar> const* params;       ///< OGC parameters
    State<TScalar, TIndex>* state;       ///< OGC state for the body
};

/**
 * @brief RTC collide function for dynamic edge - dynamic edge collisions.
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param userPtr User pointer
 * @param collisions Array of RTC collisions
 * @param nCollisions Number of RTC collisions
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void DynamicEdgeEdgeRTCCollideFunc(
    void* userPtr,
    struct RTCCollision* collisions,
    unsigned int nCollisions)
{
    auto* data = static_cast<EdgeEdgeRTCCollideFuncParams<TScalar, TIndex>*>(userPtr);
    Input<TScalar, TIndex> const* input                 = data->input;
    Params<TScalar> const* params                       = data->params;
    State<TScalar, TIndex>* state                       = data->state;
    auto const& X                                       = input->X.value();
    auto const& F                                       = input->F.value();
    auto const& E                                       = input->E.value();
    auto const& VP                                      = input->VP.value();
    auto const& EP                                      = input->EP.value();
    auto const& GVHEp                                   = input->GVHEp.value();
    auto const& GVHEadj                                 = input->GVHEadj.value();
    auto const& EHE                                     = input->EHE.value();
    auto& mEdgeLocks                                    = state->mEdgeLocks;
    auto& dmine                                         = state->dmine;
    TScalar const r                                     = params->r;
    std::vector<std::vector<ContactFace<TIndex>>>& EOGC = state->mDynamicContactFacesOfHalfEdge;
    for (unsigned int ci = 0; ci < nCollisions; ++ci)
    {
        // Get edge-edge pair (e1, e2)
        RTCCollision const& collision      = collisions[ci];
        TIndex const be1                   = static_cast<TIndex>(collision.geomID0);
        TIndex const be2                   = static_cast<TIndex>(collision.geomID1);
        TIndex const e1                    = EP(be1) + static_cast<TIndex>(collision.primID0);
        TIndex const e2                    = EP(be2) + static_cast<TIndex>(collision.primID1);
        Eigen::Vector<TIndex, 2> const e1v = E.col(e1);
        Eigen::Vector<TIndex, 2> const e2v = E.col(e2);
        // Avoid contact with same edge or adjacent edge
        bool const bIsSameEdgeOrAreAdjacent =
            e1v(0) == e2v(0) or e1v(0) == e2v(1) or e1v(1) == e2v(0) or e1v(1) == e2v(1);
        if (bIsSameEdgeOrAreAdjacent)
            continue;
        // Compute distance between edges e1 and e2 via closest point projection
        Eigen::Matrix<TScalar, 3, 2> const xe1 = X(Eigen::placeholders::all, e1v);
        Eigen::Matrix<TScalar, 3, 2> const xe2 = X(Eigen::placeholders::all, e2v);
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::SVector;
        // s,t in [0,1] are the barycentric coordinates of the closest points on edges e1 and e2
        SVector<TScalar, 2> const st = geometry::ClosestPointQueries::LineSegments(
            FromEigen(xe1.col(0)),
            FromEigen(xe1.col(1)),
            FromEigen(xe2.col(0)),
            FromEigen(xe2.col(1)));
        // Closest points on edges e1 and e2
        Eigen::Vector<TScalar, 3> const xc1 =
            (TScalar(1) - st(0)) * xe1.col(0) + st(0) * xe1.col(1);
        Eigen::Vector<TScalar, 3> const xc2 =
            (TScalar(1) - st(1)) * xe2.col(0) + st(1) * xe2.col(1);
        TScalar const d2 = (xc1 - xc2).squaredNorm();
        // Update half-edge displacement bounds
        Eigen::Vector<TIndex, 2> const ehe1 = EHE.col(e1);
        Eigen::Vector<TIndex, 2> const ehe2 = EHE.col(e2);
        common::AtomicMin(dmine(ehe1(0)), d2);
        if (ehe1(1) >= 0) // Boundary edge has no 2nd half-edge
            common::AtomicMin(dmine(ehe1(1)), d2);
        common::AtomicMin(dmine(ehe2(0)), d2);
        if (ehe2(1) >= 0) // Boundary edge has no 2nd half-edge
            common::AtomicMin(dmine(ehe2(1)), d2);
        // No contact if outside contact radius
        bool const bInContactRadius = (d2 < r * r);
        if (not bInContactRadius)
            continue;
        // Determine faces (vertex or edge) closest to edges e1 and e2, i.e. faces of xc1 and xc2
        auto const [a1, eFace1, a2, eFace2] =
            ClosestFaceEdgeToEdge(st(0), st(1), e1, e2, {e1v(0), e1v(1)}, {e2v(0), e2v(1)});
        // Synchronized updates to edges e1 and e2's contact sets
        common::AtomicExecute(mEdgeLocks(e1), [&]() {
            auto const fUpdateContactSets = [&]() {
                // If we're contacting the interior of edge e2, store the contact pairs (ehe1(0),
                // ehe2(0)) and (ehe1(1), ehe2(0)). Because we only store vertex-to-half-edge
                // adjacencies, rather than vertex-(undirected-)edge adjacencies, we store 2 contact
                // pairs for edge e1 corresponding to both of its half-edges. This way, both
                // vertices/endpoints of edge e1 can reach half-edge ehe2(0). The vertices do not
                // need to reach ehe2(1), as it is redundant/unnecessary for computing an edge-edge
                // contact potential (we only need the 2 pairs of 2 vertices in no particular
                // order).
                bool const bIsA2Edge = (eFace2 == 0);
                TIndex const a       = bIsA2Edge * ehe2(0) + (not bIsA2Edge) * a2;
                EOGC[ehe1(0)].emplace_back(a, eFace2);
                if (ehe1(1) >= 0)
                    EOGC[ehe1(1)].emplace_back(a, eFace2);
            };
            switch (static_cast<EEdgeEdgeClosestFaceType>(eFace2))
            {
                case EEdgeEdgeClosestFaceType::Vertex: {
                    if (IsVertexFeasible(X, F, GVHEp, GVHEadj, xc1, a2))
                        fUpdateContactSets();
                    break;
                }
                default /* edge */: {
                    fUpdateContactSets();
                    break;
                }
            }
        });
        common::AtomicExecute(mEdgeLocks(e2), [&]() {
            auto const fUpdateContactSets = [&]() {
                // See comment in edge e1's update above for explanation. We flip the roles of e1
                // and e2 here.
                bool const bIsA1Edge = (eFace1 == 0);
                TIndex const a       = bIsA1Edge * ehe1(0) + (not bIsA1Edge) * a1;
                EOGC[ehe2(0)].emplace_back(a, eFace1);
                if (ehe2(1) >= 0)
                    EOGC[ehe2(1)].emplace_back(a, eFace1);
            };
            switch (static_cast<EEdgeEdgeClosestFaceType>(eFace1))
            {
                case EEdgeEdgeClosestFaceType::Vertex: {
                    if (IsVertexFeasible(X, F, GVHEp, GVHEadj, xc2, a1))
                        fUpdateContactSets();
                    break;
                }
                default /* edge */: {
                    fUpdateContactSets();
                    break;
                }
            }
        });
    }
}

/**
 * @brief RTC collide function for dynamic edge - static edge collisions.
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param userPtr User pointer
 * @param collisions Array of RTC collisions
 * @param nCollisions Number of RTC collisions
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void DynamicEdgeStaticEdgeRTCCollideFunc(
    void* userPtr,
    struct RTCCollision* collisions,
    unsigned int nCollisions)
{
    auto* data = static_cast<EdgeEdgeRTCCollideFuncParams<TScalar, TIndex>*>(userPtr);
    Input<TScalar, TIndex> const* input                 = data->input;
    Params<TScalar> const* params                       = data->params;
    State<TScalar, TIndex>* state                       = data->state;
    auto const& Xenv                                    = input->Venv.value();
    auto const& Eenv                                    = input->Eenv.value();
    auto const& Fenv                                    = input->Fenv.value();
    auto const& GVHEenvp                                = input->GVHEenvp.value();
    auto const& GVHEenvadj                              = input->GVHEenvadj.value();
    auto const& EHEenv                                  = input->EHEenv.value();
    auto const& X                                       = input->X.value();
    auto const& E                                       = input->E.value();
    auto const& EHE                                     = input->EHE.value();
    auto& mEdgeLocks                                    = state->mEdgeLocks;
    auto& dmine                                         = state->dmine;
    TScalar const r                                     = params->r;
    std::vector<std::vector<ContactFace<TIndex>>>& EOGC = state->mStaticContactFacesOfHalfEdge;
    for (unsigned int ci = 0; ci < nCollisions; ++ci)
    {
        // Get edge-edge pair (e1, e2)
        RTCCollision const& collision      = collisions[ci];
        TIndex const be1                   = static_cast<TIndex>(collision.geomID0);
        TIndex const e1                    = input->DynamicEdge(be1, collision.primID0);
        TIndex const e2                    = static_cast<TIndex>(collision.primID1);
        Eigen::Vector<TIndex, 2> const e1v = E.col(e1);
        Eigen::Vector<TIndex, 2> const e2v = Eenv.col(e2);
        // Compute distance between edges e1 and e2 via closest point projection
        Eigen::Matrix<TScalar, 3, 2> const xe1 = X(Eigen::placeholders::all, e1v);
        Eigen::Matrix<TScalar, 3, 2> const xe2 = Xenv(Eigen::placeholders::all, e2v);
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::SVector;
        // s,t in [0,1] are the barycentric coordinates of the closest points on edges e1 and e2
        SVector<TScalar, 2> const st = geometry::ClosestPointQueries::LineSegments(
            FromEigen(xe1.col(0)),
            FromEigen(xe1.col(1)),
            FromEigen(xe2.col(0)),
            FromEigen(xe2.col(1)));
        // Closest points on edges e1 and e2
        Eigen::Vector<TScalar, 3> const xc1 =
            (TScalar(1) - st(0)) * xe1.col(0) + st(0) * xe1.col(1);
        Eigen::Vector<TScalar, 3> const xc2 =
            (TScalar(1) - st(1)) * xe2.col(0) + st(1) * xe2.col(1);
        TScalar const d2 = (xc1 - xc2).squaredNorm();
        // Update half-edge displacement bounds
        Eigen::Vector<TIndex, 2> const ehe1 = EHE.col(e1);
        Eigen::Vector<TIndex, 2> const ehe2 = EHEenv.col(e2);
        common::AtomicMin(dmine(ehe1(0)), d2);
        if (ehe1(1) >= 0) // Boundary edge has no 2nd half-edge
            common::AtomicMin(dmine(ehe1(1)), d2);
        // No contact if outside contact radius
        bool const bInContactRadius = (d2 < r * r);
        if (not bInContactRadius)
            continue;
        // Determine faces (vertex or edge) closest to edges e1 and e2, i.e. faces of xc1 and xc2
        auto const [a1, eFace1, a2, eFace2] =
            ClosestFaceEdgeToEdge(st(0), st(1), e1, e2, {e1v(0), e1v(1)}, {e2v(0), e2v(1)});
        // Synchronized updates to edges e1's contact sets
        common::AtomicExecute(mEdgeLocks(e1), [&]() {
            auto const fUpdateContactSets = [&]() {
                // If we're contacting the interior of edge e2, store the contact pairs (ehe1(0),
                // ehe2(0)) and (ehe1(1), ehe2(0)). Because we only store vertex-to-half-edge
                // adjacencies, rather than vertex-(undirected-)edge adjacencies, we store 2 contact
                // pairs for edge e1 corresponding to both of its half-edges. This way, both
                // vertices/endpoints of edge e1 can reach half-edge ehe2(0). The vertices do not
                // need to reach ehe2(1), as it is redundant/unnecessary for computing an edge-edge
                // contact potential (we only need the 2 pairs of 2 vertices in no particular
                // order).
                bool const bIsA2Edge = (eFace2 == 0);
                TIndex const a       = bIsA2Edge * ehe2(0) + (not bIsA2Edge) * a2;
                EOGC[ehe1(0)].emplace_back(a, eFace2);
                if (ehe1(1) >= 0)
                    EOGC[ehe1(1)].emplace_back(a, eFace2);
            };
            switch (static_cast<EEdgeEdgeClosestFaceType>(eFace2))
            {
                case EEdgeEdgeClosestFaceType::Vertex: {
                    if (IsVertexFeasible(Xenv, Fenv, GVHEenvp, GVHEenvadj, xc1, a2))
                        fUpdateContactSets();
                    break;
                }
                default /* edge */: {
                    fUpdateContactSets();
                    break;
                }
            }
        });
    }
}

} // namespace detail

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void VertexFacetContactDetection(
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params,
    State<TScalar, TIndex>& state)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.ogc.VertexFacetContactDetection");
    // Compute vertex-facet contact sets (with duplicates)
    detail::VertexFacetRTCCollideFuncParams<TScalar, TIndex> rtcCollideFuncParams{
        std::addressof(input),
        std::addressof(params),
        std::addressof(state)};
    rtcCollide(
        state.mDynamicVertexScene,
        state.mDynamicFacetScene,
        detail::DynamicVertexFacetRTCCollideFunc<TScalar, TIndex>,
        static_cast<void*>(&rtcCollideFuncParams));
    // De-duplicate contact vertices of triangles
    tbb::parallel_for(TIndex(0), TIndex(input.F->cols()), [&](TIndex f) {
        auto& vogcf = state.mDynamicContactVerticesOfTriangle[f];
        std::sort(vogcf.begin(), vogcf.end());
        vogcf.erase(std::unique(vogcf.begin(), vogcf.end()), vogcf.end());
    });
    if (input.HasStaticGeometry())
    {
        // Compute vertex-facet contact sets (with duplicates) on static geometry
        rtcCollide(
            state.mDynamicVertexScene,
            state.mStaticFacetScene,
            detail::DynamicVertexStaticFacetRTCCollideFunc<TScalar, TIndex>,
            static_cast<void*>(&rtcCollideFuncParams));
        rtcCollide(
            state.mStaticVertexScene,
            state.mDynamicFacetScene,
            detail::StaticVertexDynamicFacetRTCCollideFunc<TScalar, TIndex>,
            static_cast<void*>(&rtcCollideFuncParams));
    }
    // Finalize per-vertex and per-face displacement bounds
    state.dminv.noalias() = state.dminv.cwiseSqrt();
    state.dminf.noalias() = state.dminf.cwiseSqrt();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void EdgeEdgeContactDetection(
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params,
    State<TScalar, TIndex>& state)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.ogc.EdgeEdgeContactDetection");
    // Compute edge-edge contact sets (with duplicates) on dynamic geometry
    detail::EdgeEdgeRTCCollideFuncParams<TScalar, TIndex> rtcCollideFuncParams{
        std::addressof(input),
        std::addressof(params),
        std::addressof(state)};
    rtcCollide(
        state.mDynamicEdgeScene,
        state.mDynamicEdgeScene,
        detail::DynamicEdgeEdgeRTCCollideFunc<TScalar, TIndex>,
        static_cast<void*>(&rtcCollideFuncParams));
    // De-duplicate contact faces of half-edges on dynamic geometry
    auto const nHalfEdges = 3 * input.F->cols();
    tbb::parallel_for(TIndex(0), TIndex(nHalfEdges), [&](TIndex he) {
        auto& eogc = state.mDynamicContactFacesOfHalfEdge[he];
        std::sort(eogc.begin(), eogc.end());
        eogc.erase(std::unique(eogc.begin(), eogc.end()), eogc.end());
    });
    if (input.HasStaticGeometry())
    {
        // Compute edge-edge contact sets (with duplicates) on static geometry
        rtcCollide(
            state.mDynamicEdgeScene,
            state.mStaticEdgeScene,
            detail::DynamicEdgeStaticEdgeRTCCollideFunc<TScalar, TIndex>,
            static_cast<void*>(&rtcCollideFuncParams));
        // De-duplicate contact faces of half-edges on static geometry
        tbb::parallel_for(TIndex(0), TIndex(nHalfEdges), [&](TIndex he) {
            auto& eogc = state.mStaticContactFacesOfHalfEdge[he];
            std::sort(eogc.begin(), eogc.end());
            eogc.erase(std::unique(eogc.begin(), eogc.end()), eogc.end());
        });
    }
    // Finalize per-half-edge displacement bounds
    state.dmine.noalias() = state.dmine.cwiseSqrt();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void UpdateDisplacementBounds(
    Input<TScalar, TIndex> const& input,
    Params<TScalar> const& params,
    State<TScalar, TIndex>& state)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.ogc.UpdateDisplacementBounds");
    using ScalarType = TScalar;
    using IndexType  = TIndex;
    // dminv, dminf, dmine already computed during contact detection
    auto const& V             = input.V.value();
    auto const& GVHEp         = input.GVHEp.value();
    auto const& GVHEadj       = input.GVHEadj.value();
    IndexType const nVertices = static_cast<IndexType>(V.size());
    tbb::parallel_for(IndexType{0}, nVertices, [&](IndexType v) {
        IndexType i  = V(v);
        state.bv(v)  = state.dminv(v);
        auto hebegin = GVHEp(i);
        auto heend   = GVHEp(i + 1);
        for (IndexType k = hebegin; k < heend; ++k)
        {
            IndexType const he = GVHEadj(k);
            IndexType const f  = geometry::FaceOfHalfEdge(he);
            state.bv(v)        = std::min({state.bv(v), state.dmine(he), state.dminf(f)});
        }
        state.bv(v) *= params.gammap;
    });
}

template <common::CFloatingPoint TScalar>
std::pair<int, int> ClosestFaceFacetToVertex(TScalar u, TScalar v, TScalar w)
{
    int const nZeros     = (u == TScalar(0)) + (v == TScalar(0)) + (w == TScalar(0));
    bool const bIsVertex = (nZeros == 2);
    bool const bIsEdge   = (nZeros == 1);
    int eFace            = (bIsVertex * 2) + (bIsEdge * 1);
    int a                = bIsVertex * ((v == TScalar(1)) * 1 + (w == TScalar(1)) * 2) +
            bIsEdge * ((u == TScalar(0)) * 1 + (v == TScalar(0)) * 2);
    return {a, eFace};
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
std::tuple<int, int, int, int> ClosestFaceEdgeToEdge(
    TScalar s,
    TScalar t,
    TIndex e1,
    TIndex e2,
    std::array<TIndex, 2> e1v,
    std::array<TIndex, 2> e2v)
{
    // Vertex contact has s,t equal to 0 or 1
    bool bIsEdgeContact1 = s > TScalar(0) and s < TScalar(1);
    bool bIsEdgeContact2 = t > TScalar(0) and t < TScalar(1);
    // 0 -> edge, 1 -> vertex
    int eFace1 = (not bIsEdgeContact1) * 1;
    int eFace2 = (not bIsEdgeContact2) * 1;
    int a1     = bIsEdgeContact1 * e1 +
             (not bIsEdgeContact1) * ((s == TScalar(0)) * e1v[0] + (s == TScalar(1)) * e1v[1]);
    int a2 = bIsEdgeContact2 * e2 +
             (not bIsEdgeContact2) * ((t == TScalar(0)) * e2v[0] + (t == TScalar(1)) * e2v[1]);
    return {a1, eFace1, a2, eFace2};
}

template <common::CIndex TIndex, class TDerivedF>
TIndex
VertexFacetContactFaceIndex(Eigen::DenseBase<TDerivedF> const& F, TIndex f, int alocal, int eFace)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    return (eFace == 0) * f /* face contact, return face index */ +
           (eFace == 1) * (3 * f + alocal) /* edge contact, return half-edge index */
           + (eFace == 2) * F(alocal, f) /* vertex contact, return global point index */;
}

template <
    common::CFloatingPoint TScalar,
    common::CIndex TIndex,
    class TDerivedX,
    class TDerivedF,
    class TDerivedGVHEp,
    class TDerivedGVHEadj>
bool IsVertexFeasible(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedF> const& F,
    Eigen::DenseBase<TDerivedGVHEp> const& GVHEp,
    Eigen::DenseBase<TDerivedGVHEadj> const& GVHEadj,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex i)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    bool bInVertexFeasibleRegion{true};
    TIndex const hebegin               = GVHEp(i);
    TIndex const heend                 = GVHEp(i + 1);
    Eigen::Vector<TScalar, 3> const xv = X.col(i);
    for (TIndex he : GVHEadj(Eigen::seq(hebegin, heend - 1)))
    {
        TIndex const vp = geometry::OutgoingVertex(F, he);
        auto const xvp  = X.col(vp);
        bInVertexFeasibleRegion &= ((x - xv).dot(xv - xvp) >= TScalar(0));
    }
    return bInVertexFeasibleRegion;
}

template <
    common::CFloatingPoint TScalar,
    common::CIndex TIndex,
    class TDerivedX,
    class TDerivedF,
    class TDerivedGHEF>
bool IsEdgeFeasible(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedF> const& F,
    Eigen::DenseBase<TDerivedGHEF> const& GHEF,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex fi,
    TIndex he)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    TIndex fj = GHEF(1, he);
    // Handle boundary edge case: no adjacent face (i.e. fj == -1)
    fj       = (fj < 0) * fi + (fj >= 0) * fj;
    TIndex i = geometry::IncomingVertex(F, he);
    TIndex j = geometry::OutgoingVertex(F, he);
    TIndex k = geometry::OutgoingVertex(F, he, 1 /* step */);
    // Get the third vertex l of triangle fj that is not part of undirected edge (i,j).
    // NOTE: Whenever fj == fi (i.e. boundary edge), l == k.
    TIndex l = (F(0, fj) != i and F(0, fj) != j) * F(0, fj) +
               (F(1, fj) != i and F(1, fj) != j) * F(1, fj) +
               (F(2, fj) != i and F(2, fj) != j) * F(2, fj);
    Eigen::Vector<TScalar, 3> xi  = X.col(i);
    Eigen::Vector<TScalar, 3> xj  = X.col(j);
    Eigen::Vector<TScalar, 3> xk  = X.col(k);
    Eigen::Vector<TScalar, 3> xl  = X.col(l);
    Eigen::Vector<TScalar, 3> xij = xj - xi;
    TScalar xijn2                 = xij.squaredNorm();
    // Tangent to the plane spanned by triangle fi, perpendicular to edge (i,j)
    Eigen::Vector<TScalar, 3> pin = (xi - xk) + (xk - xi).dot(xij) / xijn2 * xij;
    // Tangent to the plane spanned by triangle fj, perpendicular to edge (i,j)
    // NOTE: whenever fj == fi (i.e. boundary edge), pjn == pin
    Eigen::Vector<TScalar, 3> pjn = (xi - xl) + (xl - xi).dot(xij) / xijn2 * xij;
    bool bInEdgeFeasibleRegion =
        ((x - xi).dot(xj - xi) >= TScalar(0)) and // within half-plane of vertex i
        ((x - xj).dot(xi - xj) >= TScalar(0)) and // within half-plane of vertex j
        ((x - xi).dot(pin) >= TScalar(0)) and     // within half-plane perpendicular to fi
        ((x - xi).dot(pjn) >= TScalar(0));        // within half-plane perpendicular to fj
    return bInEdgeFeasibleRegion;
}

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_OGC_H
