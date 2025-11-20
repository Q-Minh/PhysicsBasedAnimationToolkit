/**
 * @file MeshDynamics.h
 * @author Quoc-Minh Ton-That (tonthat@gmail.com)
 * @brief Mesh contact dynamics
 * @version 0.1
 * @date 2025-11-12
 * @copyright Copyright (c) 2025
 */

#ifndef PBAT_SIM_CONTACT_MESHDYNAMICS_H
#define PBAT_SIM_CONTACT_MESHDYNAMICS_H

#include "MeshSdfContact.h"
#include "MultiMesh.h"
#include "OffsetGeometryContact.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/geometry/Device.h"
#include "pbat/geometry/sdf/Forest.h"

#include <Eigen/Core>
#include <limits>
#include <vector>

namespace pbat::sim::contact {

/**
 * @brief Mesh contact dynamics
 *
 * We use an augmented Lagrangian for the mesh-SDF contact constraints.
 *
 * The augmented Lagrangian is written as
 * \f[
 * \sum_{c=1}^{|C|} \left[ \frac{1}{2} k_c C_c(x)^2 - \lambda_c C_c(x) \right]
 * \f]
 * where \f$ |C| \f$ is the number of constraints, \f$ C_c(x) \f$ is the constraint value for
 * constraint \f$ c \f$, \f$ k_c \f$ is its contact stiffness, and \f$ \lambda_c \f$ is its
 * Lagrange multiplier.
 *
 * The augmented Lagrangian's gradient is thus
 * \f[
 * \sum_{c=1}^{|C|} \left[ k_c C_c(x) - \lambda_c \right] \nabla C_c(x)
 * \f]
 *
 * Its hessian is
 * \f[
 * \sum_{c=1}^{|C|} \left[ k_c \nabla C_c(x) \nabla C_c(x)^T + \left( k_c C_c(x) - \lambda_c
 * \right) \nabla_x^2 C_c(x) \right]
 * \f]
 *
 * Since we use the linear contact constraints
 * \f[
 * \begin{bmatrix}
 * C_n(x) \\ C_t(x) \\ C_b(x)
 * \end{bmatrix}
 * =
 * \begin{bmatrix}
 * \mathbf{n} & \mathbf{t} & \mathbf{b}
 * \end{bmatrix}^T
 * \left( x - o \right)
 * \f] ,
 * where \f$ \mathbf{n}, \mathbf{t}, \mathbf{b} \f$ are the contact basis' normal, tangent, and
 * bitangent, and \f$ o \f$ is the contact basis' origin,
 * there is no constraint hessian \f$ \nabla_x^2 C_c(x) \f$.
 *
 * It has been shown \cite giles_augmented_2025 that warm-starting the Lagrange multipliers and
 * contact stiffnesses drastically improves constraint satisfaction. However, book-keeping of
 * cached contact constraints across time steps required to achieve such temporal coherency is
 * cumbersome. We instead store sums of Lagrange multiplier estimates on mesh features (triangles,
 * edges, vertices) that capture the total contact force affecting each feature. Per-constraint
 * Lagrange multipliers are then recovered dynamically by weighing constraint error against total
 * constraint error on the feature. The constraint stiffness is similarly uniform on each mesh
 * feature, but distinct per normal, tangent and bitangent directions.
 *
 */
class MeshDynamics
{
  public:
    using ScalarType = Scalar;
    using IndexType  = Index;

    /**
     * @brief Set the static geometry
     * @param sdfForest SDF static geometry storage
     */
    PBAT_API void SetStaticGeometry(geometry::sdf::Forest<ScalarType> sdfForest);
    /**
     * @brief Set the dynamic geometry
     * @param meshes Mesh contact geometry representation
     */
    PBAT_API void SetDynamicGeometry(MultiMesh<IndexType> meshes);
    /**
     * @brief Allocate data structures for environment contact detection
     */
    PBAT_API void AllocateEnvironmentContactDataStructures();
    /**
     * @brief Set the contact geometries
     * @param meshes Mesh contact geometry representation
     * @param sdfForest SDF static geometry storage
     * @post All data structures for contact detection are initialized, but not the algorithms (OGC
     * and MeshSDF)
     */
    void Construct(MultiMesh<IndexType> meshes, geometry::sdf::Forest<ScalarType> sdfForest);
    /**
     * @brief Initialize mesh-mesh contact detection
     *
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param device Spatial acceleration device
     * @pre `mMeshes` is set
     */
    PBAT_API void InitializeMeshMeshContactDetection(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        geometry::Device const& device);
    /**
     * @brief Initialize mesh-SDF contact detection
     * @pre `mMeshes` is set
     */
    PBAT_API void InitializeMeshEnvironmentContactDetection();
    /**
     * @brief Reformulates mesh-SDF contact constraints, i.e. their bases and origins, correctly
     * zeroing out inactive constraints and their associated Lagrange multipliers.
     * @note This does NOT evaluate the constraints, since this is typically during a dynamics
     * solve.
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     */
    PBAT_API void UpdateEnvironmentContactConstraints(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X);
    /**
     * @brief Prepares Lagrange multiplier estimates and contact stiffnesses for solver dual
     * iteration
     */
    PBAT_API void PrepareEnvironmentContactsForDualIteration();
    /**
     * @brief Updates Lagrange multiplier estimates and contact stiffnesses for mesh-SDF contact
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     */
    PBAT_API void DualUpdateEnvironmentContacts(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X);
    /**
     * @brief Recomputes geometric quantities (triangle, half-edge, and vertex areas) from current
     * positions
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     */
    PBAT_API void UpdateGeometricQuantities(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X);

    /**
     * @brief Mesh-SDF contact dynamics
     */

    /**
     * @brief 3-tuple of environment (normal, tangent, bitangent) contact constraints
     *
     * The constraint is defined as
     * \f[
     * \begin{bmatrix}
     * C_n(x) \\ C_t(x) \\ C_b(x)
     * \end{bmatrix}
     * =
     * \begin{bmatrix}
     * \mathbf{n} & \mathbf{t} & \mathbf{b}
     * \end{bmatrix}^T
     * \left( x - o \right)
     * \f]
     * where \f$ \mathbf{n}, \mathbf{t}, \mathbf{b} \f$ are the contact basis' normal, tangent, and
     * bitangent, and \f$ o \f$ is the contact basis' origin, while \f$ x \f$ is the contact point.
     *
     * TODO: Write documentation/derivation based on my personal "Linear constraint derivatives"
     * notes
     */
    struct EnvironmentContact
    {
        Eigen::Vector<ScalarType, 3> O;    ///< Contact basis origin
        Eigen::Matrix<ScalarType, 3, 3> B; ///< Contact basis (columns: normal, tangent, bitangent)
        ScalarType C;                      ///< Contact constraint values (normal)
        ScalarType lambda;                 ///< Contact Lagrange multiplier estimate (normal)
        ScalarType k;                      ///< Contact stiffness (normal)
        ScalarType mu;                     ///< Friction coefficient
        /**
         * @brief Evaluates and updates the contact constraint based on the current contact frame
         * @param xc Current contact point position
         */
        void Eval(Eigen::Vector<ScalarType, 3> const& xc);
        /**
         * @brief Estimate each contact/constraint force from Lagrange multiplier
         * @param lambdaNmax Maximum normal Lagrange multiplier magnitude
         */
        ScalarType
        ForceEstimate(ScalarType lambdaNmax = std::numeric_limits<ScalarType>::max()) const;
    };
    std::vector<EnvironmentContact>
        CF; ///< `|# triangles|` mesh-SDF triangle
            ///< contact constraints, sorted by triangle index. For triangle `f` check
            ///< `MeshSdfContact::mTriangleContactMask(f)` to determine if triangle `f` is in
            ///< contact.
    std::vector<EnvironmentContact> CHE; ///< `|# half-edges|` mesh-SDF half-edge contact
                                         ///< constraints, sorted by half-edge index. For half-edge
                                         ///< `he` check `MeshSdfContact::mHalfEdgeContactMask(he)`
                                         ///< to determine if half-edge `he` is in contact.
    std::vector<EnvironmentContact>
        CV; ///< `|# vertices|` mesh-SDF vertex
            ///< contact constraints, sorted by vertex index. For vertex `v` check
            ///< `MeshSdfContact::mVertexContactMask(v)` to determine if vertex `v` is in contact.
    /**
     * @brief Environment contact (augmented Lagrangian) dynamics parameters
     */
    struct EnvironmentContactDynamicsParams
    {
        ScalarType mu{0.3};     ///< Friction coefficient
        ScalarType kstart{1e3}; ///< Initial contact stiffness for new contacts
        ScalarType beta{10};    ///< \cite giles_augmented_2025 multiplier of constraint error for
                                ///< stiffness update
        ScalarType gamma{
            0.99}; ///< \cite giles_augmented_2025 decay factor for
                   ///< Lagrange multiplier and stiffness initialization at time step begin
        ScalarType Fnmax{1e12}; ///< Maximum normal contact force
                                ///< density magnitude
        ScalarType kmax{1e12};  ///< Maximum contact stiffness
        ScalarType epsv;        ///< \f$ \epsilon_v \f$ is IPC's relative
                         ///< velocity threshold for static to dynamic friction's smooth transition
    };
    EnvironmentContactDynamicsParams
        mEnvContactDynamicsParams; ///< Environment contact dynamics parameters

    /**
     * @brief These geometric quantities are generally useful for contact dynamics
     */
    Eigen::Vector<ScalarType, Eigen::Dynamic> FA;  ///< `|# triangles| x 1` triangle areas
    Eigen::Vector<ScalarType, Eigen::Dynamic> HEA; ///< `|# half-edges| x 1` half-edge areas
    Eigen::Vector<ScalarType, Eigen::Dynamic> VA;  ///< `|# vertices| x 1` vertex areas

    /**
     * @brief Contact detection data structures and algorithms
     */
    MultiMesh<IndexType> mMeshes;                 ///< Dynamic geometry
    OffsetGeometryContact mOffsetGeometryContact; ///< Offset-geometry contact detection
    geometry::sdf::Forest<ScalarType> mSdfForest; ///< Static geometry representation
    geometry::sdf::Composite<ScalarType> mSdf;    ///< SDF of static geometry
    MeshSdfContact mMeshSdfContact;               ///< Mesh-SDF contact detection

  protected:
    /**
     * @brief Reconstructs the triangle contact constraints list based on previous mesh-SDF contact
     * detection sweep, and "warm-starts" quantities if possible.
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     */
    void UpdateTriangleContactConstraints(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X);
    /**
     * @brief Reconstructs the half-edge contact constraints list based on previous mesh-SDF contact
     * detection sweep, and "warm-starts" quantities if possible.
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     */
    void UpdateHalfEdgeContactConstraints(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X);
    /**
     * @brief Reconstructs the vertex contact constraints list based on previous mesh-SDF contact
     * detection sweep, and "warm-starts" quantities if possible.
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     */
    void UpdateVertexContactConstraints(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X);

  private:
};

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHDYNAMICS_H
