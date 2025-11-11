#ifndef PBAT_SIM_CONTACT_MESHSDFCONTACT_H
#define PBAT_SIM_CONTACT_MESHSDFCONTACT_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/geometry/sdf/Composite.h"
#include "pbat/geometry/sdf/Forest.h"
#include "pbat/io/Archive.h"

#include <Eigen/Core>
#include <vector>

namespace pbat::sim::contact {

class MeshSdfContact
{
  public:
    using ScalarType = Scalar;
    using IndexType  = Index;
    /**
     * @brief Default constructor
     */
    PBAT_API MeshSdfContact() = default;
    /**
     * @brief Construct a new Mesh Sdf Contact object
     *
     * @param forest SDF forest representation
     * @param nTriangles Number of triangles of the colliding mesh
     */
    PBAT_API MeshSdfContact(Index nTriangles);
    /**
     * @brief Perform triangle-SDF contact detection
     *
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param sdf Environment (SDF) geometry
     */
    PBAT_API void TriangleSdfContactDetection(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        geometry::sdf::Composite<ScalarType> const& sdf);
    /**
     * @brief Serialize the mesh-SDF contact to an archive.
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the mesh-SDF contact from an archive.
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive& archive);

  private:
    // TODO: Optimize data structure used to store contact points
    std::vector<std::vector<Eigen::Vector<ScalarType, 3>>>
        mTriangleSdfContacts; ///< `|# triangles|` triangle-SDF contact points per triangle
};

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHSDFCONTACT_H
