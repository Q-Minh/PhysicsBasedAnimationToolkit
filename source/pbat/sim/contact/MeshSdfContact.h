#ifndef PBAT_SIM_CONTACT_MESHSDFCONTACT_H
#define PBAT_SIM_CONTACT_MESHSDFCONTACT_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/geometry/sdf/Composite.h"
#include "pbat/io/Archive.h"

#include <Eigen/Core>
#include <vector>

namespace pbat::sim::contact {

/**
 * @brief Parameters for mesh-SDF contact detection
 */
struct MeshSdfContactParams
{
    using ScalarType = Scalar;
    ScalarType sigmaR{0.1};   ///< Multiple of triangle size used to scale the initial trust region
                              ///< radius as \f$ \Delta_0 = \sigma_R |T| \f$, where \f$ |T| \f$ is a
                              ///< measure of the triangle size.
    ScalarType sigmaB{0.1};   ///< Multiple of triangle size used to scale the initial hessian
                              ///< approximation as \f$ \mathbf{B}_0 = \sigma_B |T| \mathbf{I} \f$,
                              ///< where \f$ |T| \f$ is a measure of the triangle size.
    ScalarType tauAred{1e-4}; ///< Proportion of triangle size at/below which we consider a
                              ///< distance reduction to be small. Must be > 0.
    ScalarType tauPred{1e-2}; ///< Proportion of triangle size at/below which we consider a
                              ///< predicted distance reduction to be small. Must be > 0.
    int nMaxContactsPerTriangle{4}; ///< Maximum number of contact points to store per triangle
    int nMaxOptimizationIterationsPerTriangle{10}; ///< Maximum number of trust-region
                                                   ///< optimization iterations per triangle
    ScalarType coordZero{
        1e-6}; ///< Tolerance for comparing if 2 contact points are to be considered duplicates.
    ScalarType hfd{1e-4}; ///< Finite difference step size used for SDF gradient estimation.
    ScalarType r{
        1e-3}; ///< Proximity threshold for considering a triangle to be a contact candidate

    /**
     * @brief Set initialization strategy parameters
     *
     * @param _sigmaR Multiple of triangle size used to scale the initial trust region radius as \f$
     * \Delta_0 = \sigma_R |T| \f$, where \f$ |T| \f$ is a measure of the triangle size.
     * @param _sigmaB Multiple of triangle size used to scale the initial hessian approximation as
     * \f$
     * \mathbf{B}_0 = \sigma_B |T| \mathbf{I} \f$, where \f$ |T| \f$ is a measure of the triangle
     * size.
     * @return Reference to this
     */
    MeshSdfContactParams& WithInitializationStrategy(ScalarType _sigmaR, ScalarType _sigmaB);
    /**
     * @brief Set termination criteria parameters
     * @param _tauAred Proportion of triangle size at/below which we consider a distance reduction
     * to be small.
     * @param _tauPred Proportion of triangle size at/below which we consider a predicted distance
     * reduction to be small.
     * @param _nMaxOptimizationIterationsPerTriangle Maximum number of trust-region optimization
     * iterations per triangle
     * @return Reference to this
     */
    MeshSdfContactParams& WithTerminationCriteria(
        ScalarType _tauAred,
        ScalarType _tauPred,
        int _nMaxOptimizationIterationsPerTriangle);
    /**
     * @brief Set contact storage limits
     * @param _nMaxContactsPerTriangle Maximum number of contact points to store per triangle
     * @return Reference to this
     */
    MeshSdfContactParams& WithContactStorageLimits(int _nMaxContactsPerTriangle);
    /**
     * @brief Set numerical parameters
     * @param _coordZero Tolerance for comparing if 2 contact points are to be considered
     * duplicates.
     * @param _hfd Finite difference step size used for SDF gradient estimation.
     * @param _r Proximity threshold for considering a triangle to be a contact candidate
     * @return Reference to this
     */
    MeshSdfContactParams&
    WithNumericalParameters(ScalarType _coordZero, ScalarType _hfd, ScalarType _r);
    /**
     * @brief Validate the parameters
     * @throw std::invalid_argument if any parameter is invalid
     * @return Reference to this
     */
    MeshSdfContactParams& Construct(bool bValidate = true);
    /**
     * @brief Serialize the mesh-SDF contact parameters to an archive.
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the mesh-SDF contact parameters from an archive.
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive& archive);
};

/**
 * @brief Mesh-SDF contact detection algorithm
 */
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
     * @param V `|# vertices| x 1` vertices (global indices into X)
     * @param F `3 x |# triangles|` triangles (global indices into X)
     */
    PBAT_API MeshSdfContact(
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F);
    /**
     * @brief Initialize the mesh-SDF contact detection
     * @param V `|# vertices| x 1` vertices (global indices into X)
     * @param F `3 x |# triangles|` triangles (global indices into X)
     */
    PBAT_API void Initialize(
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F);
    /**
     * @brief Prepare for a new iteration of contact detection
     */
    PBAT_API void PrepareIteration();
    /**
     * @brief Perform triangle-SDF contact detection
     *
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param sdf Environment (SDF) geometry
     * @pre `PrepareIteration()` has been called to reset contact data
     */
    PBAT_API void TriangleSdfContactDetection(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        geometry::sdf::Composite<ScalarType> const& sdf);
    /**
     * @brief Extract lower-dimensional contacts from triangle contacts
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param GHEF `2 x |# half edges|` half-edge to face adjacency
     * @param GXV `|# points| x 1` point to vertex adjacency s.t. `v = GXV[i]` is the vertex `v`
     * associated with point `i`
     * @post Extracted lower-dimensional contacts are stored in `mHalfEdgeContactPoints` and
     * `mVertexContactPoints`, and `mTriangleContactPoints` only contains interior points.
     */
    PBAT_API void DeduplicateContactSet(
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GXV);
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

  protected:
    /**
     * @brief Deduplicate triangle contacts
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     */
    PBAT_API void DeduplicateTriangleContacts(
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F);
    /**
     * @brief Extract lower-dimensional contacts (edge, vertex) from triangle contacts
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param GHEF `2 x |# half edges|` half-edge to face adjacency
     * @param GXV `|# points| x 1` point to vertex adjacency s.t. `v = GXV[i]` is the vertex `v`
     */
    PBAT_API void ExtractLowerDimensionalContactsFromTriangleContacts(
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GXV);
    /**
     * @brief Deduplicate half-edge contacts
     */
    PBAT_API void DeduplicateHalfEdgeContacts();

  public:
    std::vector<std::vector<Eigen::Vector<ScalarType, 2>>>
        mTriangleContactPoints; ///< `|# triangles|` per-triangle contact points in barycentric
                                ///< coordinates
    std::vector<std::vector<ScalarType>>
        mHalfEdgeContactPoints; ///< `|# half-edges|` per-half-edge contact points
                                ///< in barycentric coordinates
    Eigen::Vector<bool, Eigen::Dynamic>
        mVertexContactPoints;     ///< `|# vertices|` per-vertex contact mask
    MeshSdfContactParams mParams; ///< Mesh-SDF contact detection parameters
  private:
    Eigen::Vector<bool, Eigen::Dynamic>
        mHalfEdgeLocks; ///< `|# half-edges|` per-half-edge synchronization lock
};

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHSDFCONTACT_H
