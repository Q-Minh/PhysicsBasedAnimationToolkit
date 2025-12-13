#ifndef PBAT_SIM_CONTACT_OGC_CONTACTFACE_H
#define PBAT_SIM_CONTACT_OGC_CONTACTFACE_H

#include "pbat/common/Concepts.h"

namespace pbat::sim::contact::ogc {

/**
 * @brief Enumeration of closest face types.
 */
enum class EVertexFacetClosestFaceType : int { Vertex = 2, Edge = 1, Facet = 0 };

/**
 * @brief Enumeration of closest face types for edge-edge contacts.
 */
enum class EEdgeEdgeClosestFaceType : int { Edge = 0, Vertex = 1 };

/**
 * @brief Contact face structure.
 */
template <common::CIndex TIndex>
struct ContactFace
{
    using IndexType = TIndex; ///< Index type
    using SelfType  = ContactFace<TIndex>; ///< Self type
    /**
     * @brief Construct a new Contact Face object
     *
     * @param _a
     * @param _eFace
     */
    ContactFace(IndexType _a, IndexType _eFace) : a(_a), eFace(_eFace) {}
    /**
     * @brief Less-than operator for ordering contact faces
     * @param other
     * @return true if less than other, false otherwise
     */
    bool operator<(const SelfType& other) const
    {
        return (a < other.a) or (a == other.a and eFace < other.eFace);
    }
    /**
     * @brief Equality operator for contact faces
     * @param other
     * @return true if equal to other, false otherwise
     */
    bool operator==(const SelfType& other) const
    {
        return (a == other.a) and (eFace == other.eFace);
    }
    /**
     * @brief Get the Vertex Facet Closest Face Type enum
     * @return EVertexFacetClosestFaceType
     */
    EVertexFacetClosestFaceType VertexFacetClosestFaceType() const {
        return static_cast<EVertexFacetClosestFaceType>(eFace);
    }
    /** @brief Get the Edge Edge Closest Face Type enum
     * @return EEdgeEdgeClosestFaceType
     */
    EEdgeEdgeClosestFaceType EdgeEdgeClosestFaceType() const {
        return static_cast<EEdgeEdgeClosestFaceType>(eFace);
    }
    IndexType a; ///< Face (vertex, half-edge, edge or triangle) index
    IndexType
        eFace; ///< Face type indicator (EVertexFacetClosestFaceType | EEdgeEdgeClosestFaceType)
};

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_CONTACTFACE_H
