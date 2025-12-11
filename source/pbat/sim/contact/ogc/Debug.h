/**
 * @file Debug.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Debugging utilities for the Offset Geometry Contact (OGC) algorithm.
 * @version 0.1
 * @date 2025-12-10
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_CONTACT_OGC_DEBUG_H
#define PBAT_SIM_CONTACT_OGC_DEBUG_H

#include "ContactFace.h"
#include "Input.h"
#include "State.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/math/linalg/mini/Mini.h"

#include <numeric>
#include <tbb/parallel_for.h>
#include <vector>

namespace pbat::sim::contact::ogc {

/**
 * @brief Visual debug contact information.
 * @tparam TScalar Type of scalar
 */
template <common::CFloatingPoint TScalar>
struct VisualDebugContact
{
    Eigen::Vector<TScalar, 3> xi; ///< Contact point on feature i
    Eigen::Vector<TScalar, 3> xj; ///< Contact point on feature j
};

/**
 * @brief Converts OGC contact sets to visual debug contacts.
 *
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 * @param input Input data
 * @param state Simulation state
 * @param vertexVertexContacts Output vertex-vertex contact information
 * @param vertexEdgeContacts Output vertex-edge contact information
 * @param vertexFacetContacts Output vertex-facet contact information
 * @param edgeEdgeContacts Output edge-edge contact information
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void ToVisualDebugContacts(
    Input<TScalar, TIndex> const& input,
    State<TScalar, TIndex> const& state,
    std::vector<VisualDebugContact<TScalar>>& vertexVertexContacts,
    std::vector<VisualDebugContact<TScalar>>& vertexEdgeContacts,
    std::vector<VisualDebugContact<TScalar>>& vertexFacetContacts,
    std::vector<VisualDebugContact<TScalar>>& edgeEdgeContacts);

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void ToVisualDebugContacts(
    Input<TScalar, TIndex> const& input,
    State<TScalar, TIndex> const& state,
    std::vector<VisualDebugContact<TScalar>>& vertexVertexContacts,
    std::vector<VisualDebugContact<TScalar>>& vertexEdgeContacts,
    std::vector<VisualDebugContact<TScalar>>& vertexFacetContacts,
    std::vector<VisualDebugContact<TScalar>>& edgeEdgeContacts)
{
    std::size_t nVertexVertexContacts{0};
    std::size_t nVertexEdgeContacts{0};
    std::size_t nVertexFacetContacts{0};
    std::size_t nEdgeEdgeContacts{0};
    auto const fCountVertexFacetContacts = [&](ContactFace<TIndex> const& face) {
        switch (face.VertexFacetClosestFaceType())
        {
            case EVertexFacetClosestFaceType::Vertex: ++nVertexVertexContacts;
            case EVertexFacetClosestFaceType::Edge: ++nVertexEdgeContacts;
            case EVertexFacetClosestFaceType::Facet: ++nVertexFacetContacts;
        }
    };
    auto const fCountEdgeEdgeContacts = [&](ContactFace<TIndex> const& face) {
        switch (face.EdgeEdgeClosestFaceType())
        {
            case EEdgeEdgeClosestFaceType::Vertex: ++nVertexEdgeContacts;
            case EEdgeEdgeClosestFaceType::Edge: ++nEdgeEdgeContacts;
        }
    };
    for (auto const& facesOfVertex : state.mDynamicContactFacesOfVertex)
        for (ContactFace<TIndex> const& face : facesOfVertex)
            fCountVertexFacetContacts(face);
    for (auto const& facesOfHalfEdge : state.mDynamicContactFacesOfHalfEdge)
        for (ContactFace<TIndex> const& face : facesOfHalfEdge)
            fCountEdgeEdgeContacts(face);
    if (input.HasStaticGeometry())
    {
        for (auto const& facesOfVertex : state.mStaticContactFacesOfVertex)
            for (ContactFace<TIndex> const& face : facesOfVertex)
                fCountVertexFacetContacts(face);
        for (auto const& facesOfHalfEdge : state.mStaticContactFacesOfHalfEdge)
            for (ContactFace<TIndex> const& face : facesOfHalfEdge)
                fCountEdgeEdgeContacts(face);
    }
    vertexVertexContacts.reserve(nVertexVertexContacts);
    vertexEdgeContacts.reserve(nVertexEdgeContacts);
    vertexFacetContacts.reserve(nVertexFacetContacts);
    edgeEdgeContacts.reserve(nEdgeEdgeContacts);
    auto const& X                               = input.X.value();
    auto const& V                               = input.V.value();
    auto const& F                               = input.F.value();
    auto const fCreateDynamicVertexFacetContact = [&](TIndex vi, ContactFace<TIndex> const& face) {
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::ToEigen;
        using math::linalg::mini::SVector;
        VisualDebugContact<TScalar> contact;
        contact.xi = X.col(V(vi));
        switch (face.VertexFacetClosestFaceType())
        {
            case EVertexFacetClosestFaceType::Vertex: {
                TIndex vj  = face.a;
                contact.xj = X.col(V(vj));
                vertexVertexContacts.emplace_back(contact);
                break;
            }
            case EVertexFacetClosestFaceType::Edge: {
                TIndex const he        = face.a;
                auto const P           = X.col(geometry::IncomingVertex(F, he)).head<3>();
                auto const Q           = X.col(geometry::OutgoingVertex(F, he)).head<3>();
                SVector<TScalar, 3> xj = geometry::ClosestPointQueries::PointOnLineSegment(
                    FromEigen(contact.xi),
                    FromEigen(P),
                    FromEigen(Q));
                contact.xj = ToEigen(xj);
                vertexEdgeContacts.emplace_back(contact);
                break;
            }
            case EVertexFacetClosestFaceType::Facet: {
                TIndex const f         = face.a;
                SVector<TScalar, 3> xj = geometry::ClosestPointQueries::PointInTriangle(
                    FromEigen(contact.xi),
                    FromEigen(X.col(F(0, f)).head<3>()),
                    FromEigen(X.col(F(1, f)).head<3>()),
                    FromEigen(X.col(F(2, f)).head<3>()));
                contact.xj = ToEigen(xj);
                vertexFacetContacts.emplace_back(contact);
                break;
            }
        }
    };
    auto const fCreateDynamicEdgeEdgeContact = [&](TIndex hei, ContactFace<TIndex> const& face) {
        using math::linalg::mini::FromEigen;
        using math::linalg::mini::ToEigen;
        using math::linalg::mini::SVector;
        VisualDebugContact<TScalar> contact;
        auto const PI = X.col(geometry::IncomingVertex(F, hei)).head<3>();
        auto const QI = X.col(geometry::OutgoingVertex(F, hei)).head<3>();
        switch (face.EdgeEdgeClosestFaceType())
        {
            case EEdgeEdgeClosestFaceType::Vertex: {
                TIndex vj  = face.a;
                contact.xi = ToEigen(
                    geometry::ClosestPointQueries::PointOnLineSegment(
                        FromEigen(X.col(V(vj))),
                        FromEigen(PI),
                        FromEigen(QI)));
                contact.xj = X.col(V(vj));
                edgeEdgeContacts.emplace_back(contact);
                break;
            }
            case EEdgeEdgeClosestFaceType::Edge: {
                TIndex const he2       = face.a;
                auto const PJ          = X.col(geometry::IncomingVertex(F, he2)).head<3>();
                auto const QJ          = X.col(geometry::OutgoingVertex(F, he2)).head<3>();
                SVector<TScalar, 2> st = geometry::ClosestPointQueries::LineSegments(
                    FromEigen(PI),
                    FromEigen(QI),
                    FromEigen(PJ),
                    FromEigen(QJ));
                contact.xi = (1 - st(0)) * PI + st(0) * QI;
                contact.xj = (1 - st(1)) * PJ + st(1) * QJ;
                edgeEdgeContacts.emplace_back(contact);
                break;
            }
        }
    };
    for (auto vi = 0; vi < state.mDynamicContactFacesOfVertex.size(); ++vi)
        for (ContactFace<TIndex> const& face : state.mDynamicContactFacesOfVertex[vi])
            fCreateDynamicVertexFacetContact(vi, face);
    for (auto hei = 0; hei < state.mDynamicContactFacesOfHalfEdge.size(); ++hei)
        for (ContactFace<TIndex> const& face : state.mDynamicContactFacesOfHalfEdge[hei])
            fCreateDynamicEdgeEdgeContact(hei, face);
    if (input.HasStaticGeometry())
    {
        auto const& Xenv                           = input.Venv.value();
        auto const& Eenv                           = input.Eenv.value();
        auto const& Fenv                           = input.Fenv.value();
        auto const fCreateStaticVertexFacetContact = [&](TIndex vi,
                                                         ContactFace<TIndex> const& face) {
            VisualDebugContact<TScalar> contact;
            contact.xi = X.col(V(vi));
            using math::linalg::mini::FromEigen;
            using math::linalg::mini::ToEigen;
            using math::linalg::mini::SVector;
            switch (face.VertexFacetClosestFaceType())
            {
                case EVertexFacetClosestFaceType::Vertex: {
                    TIndex vj  = face.a;
                    contact.xj = Xenv.col(vj);
                    vertexVertexContacts.emplace_back(contact);
                    break;
                }
                case EVertexFacetClosestFaceType::Edge: {
                    TIndex const e         = face.a;
                    auto const P           = Xenv.col(Eenv(0, e)).head<3>();
                    auto const Q           = Xenv.col(Eenv(1, e)).head<3>();
                    SVector<TScalar, 3> xj = geometry::ClosestPointQueries::PointOnLineSegment(
                        FromEigen(contact.xi),
                        FromEigen(P),
                        FromEigen(Q));
                    contact.xj = ToEigen(xj);
                    vertexEdgeContacts.emplace_back(contact);
                    break;
                }
                case EVertexFacetClosestFaceType::Facet: {
                    TIndex const f         = face.a;
                    SVector<TScalar, 3> xj = geometry::ClosestPointQueries::PointInTriangle(
                        FromEigen(contact.xi),
                        FromEigen(Xenv.col(Fenv(0, f)).head<3>()),
                        FromEigen(Xenv.col(Fenv(1, f)).head<3>()),
                        FromEigen(Xenv.col(Fenv(2, f)).head<3>()));
                    contact.xj = ToEigen(xj);
                    vertexFacetContacts.emplace_back(contact);
                    break;
                }
            }
        };
        auto const fCreateStaticEdgeEdgeContact = [&](TIndex hei, ContactFace<TIndex> const& face) {
            using math::linalg::mini::FromEigen;
            using math::linalg::mini::ToEigen;
            using math::linalg::mini::SVector;
            VisualDebugContact<TScalar> contact;
            auto const PI = X.col(geometry::IncomingVertex(F, hei)).head<3>();
            auto const QI = X.col(geometry::OutgoingVertex(F, hei)).head<3>();
            switch (face.EdgeEdgeClosestFaceType())
            {
                case EEdgeEdgeClosestFaceType::Vertex: {
                    TIndex vj  = face.a;
                    contact.xi = ToEigen(
                        geometry::ClosestPointQueries::PointOnLineSegment(
                            FromEigen(Xenv.col(vj)),
                            FromEigen(PI),
                            FromEigen(QI)));
                    contact.xj = Xenv.col(vj);
                    edgeEdgeContacts.emplace_back(contact);
                    break;
                }
                case EEdgeEdgeClosestFaceType::Edge: {
                    TIndex const e2        = face.a;
                    auto const PJ          = Xenv.col(Eenv(0, e2)).head<3>();
                    auto const QJ          = Xenv.col(Eenv(1, e2)).head<3>();
                    SVector<TScalar, 2> st = geometry::ClosestPointQueries::LineSegments(
                        FromEigen(PI),
                        FromEigen(QI),
                        FromEigen(PJ),
                        FromEigen(QJ));
                    contact.xi = (1 - st(0)) * PI + st(0) * QI;
                    contact.xj = (1 - st(1)) * PJ + st(1) * QJ;
                    edgeEdgeContacts.emplace_back(contact);
                    break;
                }
            }
        };
        for (auto vi = 0; vi < state.mStaticContactFacesOfVertex.size(); ++vi)
            for (ContactFace<TIndex> const& face : state.mStaticContactFacesOfVertex[vi])
                fCreateStaticVertexFacetContact(vi, face);
        for (auto hei = 0; hei < state.mStaticContactFacesOfHalfEdge.size(); ++hei)
            for (ContactFace<TIndex> const& face : state.mStaticContactFacesOfHalfEdge[hei])
                fCreateStaticEdgeEdgeContact(hei, face);
    }
}

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_DEBUG_H
