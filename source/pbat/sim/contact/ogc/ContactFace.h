#ifndef PBAT_SIM_CONTACT_OGC_CONTACTFACE_H
#define PBAT_SIM_CONTACT_OGC_CONTACTFACE_H

#include "pbat/common/Concepts.h"

namespace pbat::sim::contact::ogc {

/**
 * @brief Contact face structure.
 */
template <common::CIndex TIndex>
struct ContactFace
{
    using IndexType = TIndex;
    using SelfType  = ContactFace<TIndex>;
    /**
     * @brief Construct a new Contact Face object
     *
     * @param _a
     * @param _eFace
     */
    ContactFace(IndexType _a, IndexType _eFace) : a(_a), eFace(_eFace) {}
    /**
     * @brief Check if the contact face is a triangle
     * @return true if triangle, false otherwise
     */
    bool IsTriangle() const { return eFace == 0; }
    /**
     * @brief Check if the contact face is an edge
     * @return true if edge, false otherwise
     */
    bool IsEdge() const { return eFace == 1; }
    /**
     * @brief Check if the contact face is a vertex
     * @return true if vertex, false otherwise
     */
    bool IsVertex() const { return eFace == 2; }
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
    IndexType a;     ///< Face (vertex, half-edge, edge or triangle) index
    IndexType eFace; ///< Face type indicator: (0 | 1 | 2) -> (triangle | (half-)edge | vertex)
};

} // namespace pbat::sim::contact::ogc

#endif // PBAT_SIM_CONTACT_OGC_CONTACTFACE_H
