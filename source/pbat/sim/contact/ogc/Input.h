/**
 * @file Input.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Input for Offset Geometry Contact (OGC) algorithm.
 * @version 0.1
 * @date 2025-12-10
 * @copyright Copyright (c) 2025
 */
#ifndef PBAT_SIM_CONTACT_OGC_INPUT_H
#define PBAT_SIM_CONTACT_OGC_INPUT_H

#include "pbat/common/Concepts.h"

#include <Eigen/Core>
#include <exception>
#include <optional>

namespace pbat::sim::contact::ogc {

/**
 * @brief User data structure for the OGC internal functions.
 *
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct Input
{
    using SelfType = Input<TScalar, TIndex>;
    /**
     * @brief Borrow dynamic geometry data.
     * @param _X `3 x |# points|` dynamic vertex positions
     * @param _V `|# verts|` vertex indices into `X`
     * @param _F `3 x |# facets|` face indices into `X`
     * @param _E `2 x |# edges|` edge indices into `X`
     * @param _VP `|# bodies + 1|` prefix sum of vertex counts per body
     * @param _FP `|# bodies + 1|` prefix sum of face counts per body
     * @param _EP `|# bodies + 1|` prefix sum of edge counts per body
     * @param _GVHEp `|# points + 1|` point to half-edge prefix
     * @param _GVHEadj `|# half edges|` point to half-edge adjacency
     * @param _GHEF `2 x |# half edges|` half-edge to face adjacency
     * @param _EHE `2 x |# edges|` half-edge indices for edges
     * @return SelfType& Reference to this
     */
    SelfType& WithDynamicGeometry(
        Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& _X,
        Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _V,
        Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& _F,
        Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _E,
        Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _VP,
        Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _FP,
        Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _EP,
        Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _GVHEp,
        Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _GVHEadj,
        Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _GHEF,
        Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _EHE);
    /**
     * @brief Borrow static geometry data.
     * @param _Venv `3 x |# env. verts|` static vertex positions
     * @param _Fenv `3 x |# env. facets|` static facet indices into `Venv`
     * @param _GVHEp `|# env. verts + 1|` point to half-edge prefix
     * @param _GVHEadj `|# env. half edges|` point to half-edge adjacency
     * @param _GHEF `2 x |# env. half edges|` half-edge to face adjacency
     * @param _EHE `2 x |# env. edges|` half-edge indices for edges
     * @return SelfType& Reference to this
     */
    SelfType& WithStaticGeometry(
        Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& _Venv,
        Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _Eenv,
        Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& _Fenv,
        Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _GVHEp,
        Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _GVHEadj,
        Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _GHEF,
        Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _EHE);
    /**
     * @brief Check if dynamic geometry is provided
     * @return true if dynamic geometry is provided
     */
    bool HasDynamicGeometry() const;
    /**
     * @brief Check if static geometry is provided
     * @return true if static geometry is provided
     */
    bool HasStaticGeometry() const;
    /**
     * @brief Validate construction of the input data.
     * @return SelfType& Reference to this
     * @throws std::runtime_error if the input data is invalid
     */
    SelfType& Construct();
    /**
     * @brief Get (global) dynamic vertex index of body `b` given its local vertex index `v`
     * @param b Body index
     * @param v (Local) vertex index
     * @return TIndex Dynamic vertex index
     */
    TIndex DynamicVertex(Eigen::Index b, Eigen::Index v) const;
    /**
     * @brief Get (global) dynamic edge index of body `b` given its local edge index `e`
     * @param b Body index
     * @param e (Local) edge index
     * @return TIndex Dynamic edge index
     */
    TIndex DynamicEdge(Eigen::Index b, Eigen::Index e) const;
    /**
     * @brief Get (global) dynamic facet index of body `b` given its local facet index `f`
     * @param b Body index
     * @param f (Local) facet index
     * @return TIndex Dynamic facet index
     */
    TIndex DynamicFacet(Eigen::Index b, Eigen::Index f) const;
    /**
     * @brief Get number of bodies
     * @return Eigen::Index Number of bodies
     */
    Eigen::Index NumBodies() const;
    /**
     * @brief Get number of vertices for a dynamic body
     * @param b Body index
     * @return Eigen::Index Number of vertices for the body
     */
    Eigen::Index NumVertices(Eigen::Index b) const;
    /**
     * @brief Get number of edges for a dynamic body
     * @param b Body index
     * @return Eigen::Index Number of edges for the body
     */
    Eigen::Index NumEdges(Eigen::Index b) const;
    /**
     * @brief Get number of facets for a dynamic body
     * @param b Body index
     * @return Eigen::Index Number of facets for the body
     */
    Eigen::Index NumFacets(Eigen::Index b) const;
    /**
     * @brief Get number of static vertices
     * @return Eigen::Index Number of static vertices
     */
    Eigen::Index NumStaticVertices() const;
    /**
     * @brief Get number of static edges
     * @return Eigen::Index Number of static edges
     */
    Eigen::Index NumStaticEdges() const;
    /**
     * @brief Get number of static facets
     * @return Eigen::Index Number of static facets
     */
    Eigen::Index NumStaticFacets() const;

    /**
     * @brief Dynamic geometry
     */
    std::optional<Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const>>
        X; ///< `3 x |# points|` dynamic points
    std::optional<Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const>>
        V; ///< `|# verts|` dynamic vertex indices into `X`
    std::optional<Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const>>
        F; ///< `3 x |# facets|` dynamic facet indices into `X`
    std::optional<Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const>>
        E; ///< `2 x |# edges|` dynamic edge indices into `X`
    std::optional<Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const>>
        VP; ///< `|# bodies + 1|` dynamic vertex prefix
    std::optional<Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const>>
        FP; ///< `|# bodies + 1|` dynamic face prefix
    std::optional<Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const>>
        EP; ///< `|# bodies + 1|` dynamic edge prefix
    std::optional<Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const>>
        GVHEp; ///< `|# points + 1|` point to half-edge prefix
    std::optional<Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const>>
        GVHEadj; ///< `|# half edges|` point to half-edge adjacency
    std::optional<Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const>>
        GHEF; ///< `2 x |# half edges|` half-edge to face adjacency
    std::optional<Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const>>
        EHE; ///< `2 x |# edges|` half-edge indices for edges
    /**
     * @brief Static geometry
     */
    std::optional<Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const>>
        Venv; ///< `3 x |# env. verts|` static environment points
    std::optional<Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const>>
        Eenv; ///< `2 x |# env. edges|` static environment edge indices into `Venv`
    std::optional<Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const>>
        Fenv; ///< `3 x |# env. facets|` static environment facet indices into `Venv`
    std::optional<Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const>>
        GVHEenvp; ///< `|# env. verts + 1|` point to half-edge prefix
    std::optional<Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const>>
        GVHEenvadj; ///< `|# env. half edges|` point to half-edge adjacency
    std::optional<Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const>>
        GHEFenv; ///< `2 x |# env. half edges|` half-edge to face adjacency
    std::optional<Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const>>
        EHEenv; ///< `2 x |# env. edges|` half-edge indices for edges
};

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Input<TScalar, TIndex>& Input<TScalar, TIndex>::WithDynamicGeometry(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& _X,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _V,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& _F,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _E,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _VP,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _FP,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _EP,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _GVHEp,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _GVHEadj,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _GHEF,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _EHE)
{
    X.emplace(_X);
    V.emplace(_V);
    F.emplace(_F);
    E.emplace(_E);
    VP.emplace(_VP);
    FP.emplace(_FP);
    EP.emplace(_EP);
    GVHEp.emplace(_GVHEp);
    GVHEadj.emplace(_GVHEadj);
    GHEF.emplace(_GHEF);
    EHE.emplace(_EHE);
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Input<TScalar, TIndex>& Input<TScalar, TIndex>::WithStaticGeometry(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& _Venv,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _Eenv,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& _Fenv,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _GVHEp,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& _GVHEadj,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _GHEF,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& _EHE)
{
    Venv.emplace(_Venv);
    Eenv.emplace(_Eenv);
    Fenv.emplace(_Fenv);
    GVHEenvp.emplace(_GVHEp);
    GVHEenvadj.emplace(_GVHEadj);
    GHEFenv.emplace(_GHEF);
    EHEenv.emplace(_EHE);
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline bool Input<TScalar, TIndex>::HasDynamicGeometry() const
{
    return X.has_value() and V.has_value() and F.has_value() and E.has_value() and
           VP.has_value() and FP.has_value() and EP.has_value() and GVHEp.has_value() and
           GVHEadj.has_value() and GHEF.has_value() and EHE.has_value();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline bool Input<TScalar, TIndex>::HasStaticGeometry() const
{
    return Venv.has_value() and Eenv.has_value() and Fenv.has_value() and GVHEenvp.has_value() and
           GVHEenvadj.has_value() and GHEFenv.has_value() and EHEenv.has_value();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Input<TScalar, TIndex>& Input<TScalar, TIndex>::Construct()
{
    // Validate dynamic geometry
    if (not HasDynamicGeometry())
    {
        throw std::runtime_error("Input::Construct: Incomplete dynamic geometry data.");
    }
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline TIndex Input<TScalar, TIndex>::DynamicVertex(Eigen::Index b, Eigen::Index v) const
{
    return VP.value()(b) + v;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline TIndex Input<TScalar, TIndex>::DynamicEdge(Eigen::Index b, Eigen::Index e) const
{
    return EP.value()(b) + e;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline TIndex Input<TScalar, TIndex>::DynamicFacet(Eigen::Index b, Eigen::Index f) const
{
    return FP.value()(b) + f;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index Input<TScalar, TIndex>::NumBodies() const
{
    return VP.value().size() - 1;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index Input<TScalar, TIndex>::NumVertices(Eigen::Index b) const
{
    return VP.value()(b + 1) - VP.value()(b);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index Input<TScalar, TIndex>::NumEdges(Eigen::Index b) const
{
    return EP.value()(b + 1) - EP.value()(b);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index Input<TScalar, TIndex>::NumFacets(Eigen::Index b) const
{
    return FP.value()(b + 1) - FP.value()(b);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index Input<TScalar, TIndex>::NumStaticVertices() const
{
    return Venv.value().cols();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index Input<TScalar, TIndex>::NumStaticEdges() const
{
    return Eenv.value().cols();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index Input<TScalar, TIndex>::NumStaticFacets() const
{
    return Fenv.value().cols();
}

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_INPUT_H
