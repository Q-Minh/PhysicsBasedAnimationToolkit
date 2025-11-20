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
    int nMaxOptimizationIterations{10}; ///< Maximum number of trust-region
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
     * @param params Mesh-SDF contact detection parameters
     */
    PBAT_API MeshSdfContact(
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        MeshSdfContactParams const& params = MeshSdfContactParams());
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
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param GHEF `2 x |# half-edges|` half-edge to triangle adjacency s.t. `GHEF(0, hei) == fi`
     * and `GHEF(1, hei) == fj` where `fi` and `fj` are edge-adjacent triangles
     * @param EHE `2 x |# edges|` edge to half-edge adjacency s.t. `EHE(0, e) == hei` and `EHE(1,
     * e)` are opposite half-edges of edge `e`, with `-1` indicating no opposite half-edge. `-1`
     * may only appear in second row.
     * @param GXV `|# points| x 1` point to vertex adjacency s.t. `GXV(pi) == vi` where `vi` is the
     * vertex index of point `pi`. If a point does not correspond to a vertex, `GXV(pi) == -1`.
     * @param sdf Environment (SDF) geometry
     * @pre `PrepareIteration()` has been called to reset contact data
     */
    PBAT_API void TriangleSdfContactDetection(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GXV,
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

  public:
    Eigen::Vector<bool, Eigen::Dynamic>
        mTriangleContactMask; ///< `|# triangles|` per-triangle contact mask
    Eigen::Matrix<ScalarType, 2, Eigen::Dynamic>
        mTriangleContactPoints; ///< `2 x |# triangle contacts|` triangle contact points in
                                ///< barycentric coordinates `(1-u-v), u, v` where we only store `u,
                                ///< v`
    Eigen::Vector<bool, Eigen::Dynamic>
        mHalfEdgeContactMask; ///< `|# half-edges|` per-half-edge contact mask
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        mHalfEdgeContactPoints; ///< `|# half-edges|` per-half-edge contact points in barycentric
                                ///< coordinates `(1-u), u` where we only store `u`
    Eigen::Vector<bool, Eigen::Dynamic>
        mVertexContactMask; ///< `|# vertices|` per-vertex contact mask
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        mVertexDisplacementBounds; ///< `|# vertices|` per-vertex contact
    ///< displacement bounds for near penetration-free movement.
    MeshSdfContactParams mParams; ///< Mesh-SDF contact detection parameters
  private:
    Eigen::Vector<bool, Eigen::Dynamic>
        mVertexLocks; ///< `|# vertices|` per-vertex synchronization locks
};

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHSDFCONTACT_H
