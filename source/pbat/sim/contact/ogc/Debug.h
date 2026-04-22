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
    // DEPRECATED:
    // Update this code when we'll introduce friction into our new MeshDynamics API!
    // Friction necessarily requires local contact orthogonal frames, which are
    // what we want to be able to visualize in debug.

    // graph::DenseAdjacencySet<void, TIndex> XX, XE, XF, EE;
    // auto mXXcpy = state.mXX;
    // auto mXEcpy = state.mXE;
    // auto mXFcpy = state.mXF;
    // auto mEEcpy = state.mEE;
    // XX.Reduce(mXXcpy.begin(), mXXcpy.end());
    // XE.Reduce(mXEcpy.begin(), mXEcpy.end());
    // XF.Reduce(mXFcpy.begin(), mXFcpy.end());
    // EE.Reduce(mEEcpy.begin(), mEEcpy.end());
    // std::size_t nVertexVertexContacts = XX.Size();
    // std::size_t nVertexEdgeContacts   = XE.Size();
    // std::size_t nVertexFacetContacts  = XF.Size();
    // std::size_t nEdgeEdgeContacts     = EE.Size();
    // vertexVertexContacts.reserve(nVertexVertexContacts);
    // vertexEdgeContacts.reserve(nVertexEdgeContacts);
    // vertexFacetContacts.reserve(nVertexFacetContacts);
    // edgeEdgeContacts.reserve(nEdgeEdgeContacts);
    // auto const& X               = input.X.value();
    // auto const& E               = input.E.value();
    // auto const& V               = input.V.value();
    // auto const& F               = input.F.value();
    // auto const* Xenv            = (input.Venv) ? std::addressof(input.Venv.value()) : nullptr;
    // auto const* Eenv            = (input.Eenv) ? std::addressof(input.Eenv.value()) : nullptr;
    // auto const* Fenv            = (input.Fenv) ? std::addressof(input.Fenv.value()) : nullptr;
    // using StateType             = State<TScalar, TIndex>;
    // auto const Xprefix          = state.mPointGeometryPrefix[StateType::EGeometry::Dynamic];
    // auto const HEprefix         = state.mHalfEdgeGeometryPrefix[StateType::EGeometry::Dynamic];
    // auto const Eprefix          = state.mEdgeGeometryPrefix[StateType::EGeometry::Dynamic];
    // auto const Fprefix          = state.mTriangleGeometryPrefix[StateType::EGeometry::Dynamic];
    // auto const fCreateVVContact = [&](TIndex i, TIndex j) {
    //     VisualDebugContact<TScalar> contact;
    //     contact.xi = (i < Xprefix) ? X.col(i) : Xenv->col(i - Xprefix);
    //     contact.xj = (j < Xprefix) ? X.col(j) : Xenv->col(j - Xprefix);
    //     vertexVertexContacts.emplace_back(contact);
    // };
    // auto const fCreateVEContact = [&](TIndex i, TIndex he) {
    //     using math::linalg::mini::FromEigen;
    //     using math::linalg::mini::ToEigen;
    //     using math::linalg::mini::SVector;
    //     VisualDebugContact<TScalar> contact;
    //     contact.xi = (i < Xprefix) ? X.col(i) : Xenv->col(i - Xprefix);
    //     Eigen::Vector<TScalar, 3> const P =
    //         (he < HEprefix) ?
    //             X.col(geometry::IncomingVertex(F, he)).template head<3>() :
    //             Xenv->col(geometry::IncomingVertex(*Fenv, he - HEprefix)).template head<3>();
    //     Eigen::Vector<TScalar, 3> const Q =
    //         (he < HEprefix) ?
    //             X.col(geometry::OutgoingVertex(F, he)).template head<3>() :
    //             Xenv->col(geometry::OutgoingVertex(*Fenv, he - HEprefix)).template head<3>();
    //     SVector<TScalar, 3> xj = geometry::ClosestPointQueries::PointOnLineSegment(
    //         FromEigen(contact.xi),
    //         FromEigen(P),
    //         FromEigen(Q));
    //     contact.xj = ToEigen(xj);
    //     vertexEdgeContacts.emplace_back(contact);
    // };
    // auto const fCreateVFContact = [&](TIndex i, TIndex f) {
    //     using math::linalg::mini::FromEigen;
    //     using math::linalg::mini::ToEigen;
    //     using math::linalg::mini::SVector;
    //     VisualDebugContact<TScalar> contact;
    //     contact.xi = (i < Xprefix) ? X.col(i) : Xenv->col(i - Xprefix);
    //     Eigen::Matrix<TScalar, 3, 3> XF;
    //     XF.col(0)              = (f < Fprefix) ? X.col(F(0, f)).template head<3>() :
    //                                              Xenv->col((*Fenv)(0, f - Fprefix)).template
    //                                              head<3>();
    //     XF.col(1)              = (f < Fprefix) ? X.col(F(1, f)).template head<3>() :
    //                                              Xenv->col((*Fenv)(1, f - Fprefix)).template
    //                                              head<3>();
    //     XF.col(2)              = (f < Fprefix) ? X.col(F(2, f)).template head<3>() :
    //                                              Xenv->col((*Fenv)(2, f - Fprefix)).template
    //                                              head<3>();
    //     SVector<TScalar, 3> xj = geometry::ClosestPointQueries::PointInTriangle(
    //         FromEigen(contact.xi),
    //         FromEigen(XF.col(0)),
    //         FromEigen(XF.col(1)),
    //         FromEigen(XF.col(2)));
    //     contact.xj = ToEigen(xj);
    //     vertexFacetContacts.emplace_back(contact);
    // };
    // auto const fCreateEEContact = [&](TIndex ei, TIndex ej) {
    //     using math::linalg::mini::FromEigen;
    //     using math::linalg::mini::ToEigen;
    //     using math::linalg::mini::SVector;
    //     VisualDebugContact<TScalar> contact;
    //     Eigen::Vector<TScalar, 3> PI, QI, PJ, QJ;
    //     PI                     = (ei < Eprefix) ? X.col(E(0, ei)).template head<3>() :
    //                                               Xenv->col((*Eenv)(0, ei - Eprefix)).template
    //                                               head<3>();
    //     QI                     = (ei < Eprefix) ? X.col(E(1, ei)).template head<3>() :
    //                                               Xenv->col((*Eenv)(1, ei - Eprefix)).template
    //                                               head<3>();
    //     PJ                     = (ej < Eprefix) ? X.col(E(0, ej)).template head<3>() :
    //                                               Xenv->col((*Eenv)(0, ej - Eprefix)).template
    //                                               head<3>();
    //     QJ                     = (ej < Eprefix) ? X.col(E(1, ej)).template head<3>() :
    //                                               Xenv->col((*Eenv)(1, ej - Eprefix)).template
    //                                               head<3>();
    //     SVector<TScalar, 2> st = geometry::ClosestPointQueries::LineSegments(
    //         FromEigen(PI),
    //         FromEigen(QI),
    //         FromEigen(PJ),
    //         FromEigen(QJ));
    //     contact.xi = (1 - st(0)) * PI + st(0) * QI;
    //     contact.xj = (1 - st(1)) * PJ + st(1) * QJ;
    //     edgeEdgeContacts.emplace_back(contact);
    // };
    // XX.ForAll([&](TIndex i, TIndex j) { fCreateVVContact(i, j); });
    // XE.ForAll([&](TIndex i, TIndex hei) { fCreateVEContact(i, hei); });
    // XF.ForAll([&](TIndex i, TIndex fi) { fCreateVFContact(i, fi); });
    // EE.ForAll([&](TIndex ei, TIndex ej) { fCreateEEContact(ei, ej); });
}

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_DEBUG_H
