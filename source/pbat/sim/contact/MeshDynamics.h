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

#include "Constraints.h"
#include "MultiMesh.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Atomic.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/Device.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/graph/DenseAdjacencySet.h"
#include "pbat/io/Archive.h"
#include "pbat/math/linalg/FilterEigenvalues.h"
#include "pbat/math/linalg/mini/Matrix.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/contact/Friction.h"
#include "pbat/sim/contact/Potentials.h"
#include "pbat/sim/contact/ogc/Ogc.h"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <tbb/parallel_for.h>
#include <type_traits>
#include <vector>

namespace pbat::sim::contact {

/**
 * @brief Mesh contact dynamics
 * @tparam TScalar Type of scalar
 * @tparam TIndex Type of index
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
class MeshDynamics
{
  public:
    using ScalarType = TScalar; ///< Scalar type
    using IndexType  = TIndex;  ///< Index type

    /**
     * @brief Mesh dynamics parameters
     */
    struct Params
    {
        using SelfType = typename MeshDynamics<TScalar, TIndex>::Params; ///< Self type

        ogc::Params<TScalar> mOgcParams; ///< OGC parameters
        TScalar epsv{1e-3}; ///< IPC's relative velocity threshold for static to dynamic friction's
                            ///< smooth transition
        TScalar kc{1e3};    ///< OGC contact stiffness parameter, `kc > 0`
        TScalar mu{0.5};    ///< OGC friction coefficient, `mu >= 0`
        TScalar rqstart{0}; ///< Base query radius (larger than contact radius `r`) on which we add
                            ///< a linear function of inertial target distance to initialize the
                            ///< actual query radius
        TScalar betarq{1};  ///< Slope of the linear function of inertial target distance to add to
                            ///< `rqstart` to initialize the actual query radius
        bool bDeactivate{false}; ///< Whether to deactivate contacts

        /**
         * @brief Set the OGC parameters
         * @param params OGC parameters
         * @return Reference to this
         */
        SelfType& WithOgcParams(ogc::Params<TScalar> const& params);
        /**
         * @brief Set the frictional contact parameters
         *
         * @param mu Friction coefficient
         * @param epsv Relative velocity threshold for static to dynamic friction's smooth
         * transition
         * @return Reference to this
         */
        SelfType& WithFrictionalContact(TScalar mu, TScalar epsv);
        /**
         * @brief Set the normal contact parameters
         * @param kc Contact stiffness parameter, `kc > 0`
         * @return Reference to this
         */
        SelfType& WithNormalContact(TScalar kc);
        /**
         * @brief Set the query radius initialization parameters
         * @param rqstart Base query radius
         * @param betarq Slope of the linear function of inertial target distance to add to
         * `rqstart` to initialize the actual query radius
         * @return Reference to this
         */
        SelfType& WithQueryRadiusInitialization(TScalar rqstart, TScalar betarq);
        /**
         * @brief Construct the Params object
         * @param bValidate Whether to validate parameters
         * @return Reference to this
         */
        SelfType& Construct(bool bValidate = true);
        /**
         * @brief Query the contact query radius given the inertial target distance
         * @param inertialTargetDistance Inertial target distance (or other relevant distance)
         * @post `mOgcParams.rq` is set
         */
        void ComputeQueryRadius(TScalar inertialTargetDistance);
        /**
         * @brief Activate or deactivate contacts
         * @param bActive true to activate, false to deactivate
         */
        void Activate(bool bActive = true);
        /**
         * @brief Deactivate contacts
         */
        void Deactivate();
        /**
         * @brief Serialize to archive
         * @param archive Archive to serialize to
         */
        void Serialize(io::Archive& archive) const;
        /**
         * @brief Deserialize from archive
         * @param archive Archive to deserialize from
         */
        void Deserialize(io::Archive const& archive);

        /**
         * @brief Read/Write members, not part of the configuration
         */
        TScalar kcp; ///< `kcp = tau*kc*(tau - r)^2`, where `tau = r/2`
        TScalar b;   ///< `b = kc/2*(r - tau)^2 + kcp*log(tau)`, where `tau = r/2`
    };

    /**
     * @brief Construct a new Mesh Dynamics object
     */
    MeshDynamics(Params const& params = Params());
    /**
     * @brief Set the contact geometries
     * @param Xdynamic `3 x |# points|` dynamic point positions (column-major: one point per column)
     * @param dynamicMeshes Dynamic mesh contact geometry representation
     * @param Xstatic `3 x |# points|` static point positions (column-major: one point per column)
     * @param staticMeshes Static mesh contact geometry representation
     */
    void Construct(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xdynamic,
        MultiMesh<IndexType> dynamicMeshes,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xstatic,
        MultiMesh<IndexType> staticMeshes);
    /**
     * @brief Initialize the Mesh Dynamics object
     * @param device Device to use for acceleration structures
     * @pre `SetDynamicGeometry()` or `Construct()` has been called
     */
    void Initialize(geometry::Device device);
    /**
     * @brief Truncate displacements to satisfy the computed displacement bounds
     * @tparam TDerivedXkp1 Writeable matrix type
     * @param Xkp1 `3 x |# points|` or `3*|# points| x 1` proposed new point positions
     * @return Number of truncated points in this call.
     * @pre `Initialize()` has been called
     * @post Displacements have been truncated to satisfy the computed displacement bounds
     */
    template <class TDerivedXkp1>
    Eigen::Index TruncateDisplacedPositions(Eigen::MatrixBase<TDerivedXkp1>& Xkp1);
    /**
     * @brief Truncate displacements from positions to satisfy the computed displacement bounds
     * @tparam TDerivedXkp1 Writeable matrix type
     * @tparam TMask Eigen dense base s.t. TMask::Scalar is convertible to bool
     * @param Xkp1 `3 x |# points|` or `3*|# points| x 1` proposed new point positions
     * @param mask `|# points| x 1` mask of points to ignore (true = ignore, false = process)
     * @return Number of truncated points in this call.
     * @pre `Initialize()` has been called
     * @post Displacements have been truncated to satisfy the computed displacement bounds
     */
    template <class TDerivedXkp1, class TMask>
    Eigen::Index TruncateDisplacedPositions(
        Eigen::MatrixBase<TDerivedXkp1>& Xkp1,
        Eigen::DenseBase<TMask> const& mask);
    /**
     * @brief Truncate displacements to satisfy the computed displacement bounds
     * @tparam TDerivedDxkp1 Writeable matrix type
     * @tparam TMask Eigen dense base s.t. TMask::Scalar is convertible to bool
     * @param Dxkp1 `3 x |# points|` or `3*|# points| x 1` displacements
     * @param mask `|# points| x 1` mask of points to ignore (true = ignore, false = process)
     * @return Number of truncated points in this call.
     * @pre `Initialize()` has been called
     * @post Displacements have been truncated to satisfy the computed displacement bounds
     */
    template <class TDerivedDxkp1, class TMask>
    Eigen::Index TruncateDisplacements(
        Eigen::MatrixBase<TDerivedDxkp1>& Dxkp1,
        Eigen::DenseBase<TMask> const& mask);
    /**
     * @brief Request computation of displacement bounds
     */
    void RequestDisplacementBoundsComputation();
    /**
     * @brief Get the number of truncated points from the last `TruncateDisplacedPositions()`
     * call
     * @return Number of truncated points
     */
    Eigen::Index NumTruncatedPoints() const;
    /**
     * @brief Check if displacement bounds computation is required
     * @return true if displacement bounds computation is required, false otherwise
     */
    bool RequiresBoundsComputation() const;
    /**
     * @brief Executes a collision detection pass and computes resulting per-point displacement
     * bounds.
     * @tparam TDerivedX Matrix type
     * @param X `3 x |# points|` current point positions (column-major: one point per column)
     * @pre `TruncateDisplacedPositions()` has been called
     */
    template <class TDerivedX>
    void ComputeDisplacementBounds(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief For each point-(dynamic)face contact of point `i`, invoke the appropriate callback
     *
     * @tparam FOnPointPointContact Callable with signature `void(TIndex j)` for point `j`
     * @tparam FOnPointLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex, 2>
     * einds)` for edge `einds`
     * @tparam FOnPointTriangleContact Callable with signature `void(Eigen::Vector<TIndex, 3>
     * finds)` for triangle `finds`
     * @param i Point index
     * @param fOnPointPointContact Point-point contact handler
     * @param fOnPointEdgeContact Point-edge contact handler
     * @param fOnPointTriangleContact Point-triangle contact handler
     */
    template <
        class FOnPointPointContact,
        class FOnPointLineSegmentContact,
        class FOnPointTriangleContact>
    void ForEachPointDynamicMeshContact(
        IndexType i,
        FOnPointPointContact&& fOnPointPointContact,
        FOnPointLineSegmentContact&& fOnPointEdgeContact,
        FOnPointTriangleContact&& fOnPointTriangleContact);
    /**
     * @brief For each point-(static)face contact of point `i`, invoke the appropriate callback
     *
     * @tparam FOnPointPointContact Callable with signature `void(TIndex j)` for point `j`
     * @tparam FOnPointLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex, 2>
     * einds)` for edge `einds`
     * @tparam FOnPointTriangleContact Callable with signature `void(Eigen::Vector<TIndex, 3>
     * finds)` for triangle `finds`
     * @param i Point index
     * @param fOnPointPointContact Point-point contact handler
     * @param fOnPointEdgeContact Point-edge contact handler
     * @param fOnPointTriangleContact Point-triangle contact handler
     */
    template <
        class FOnPointPointContact,
        class FOnPointLineSegmentContact,
        class FOnPointTriangleContact>
    void ForEachPointStaticMeshContact(
        IndexType i,
        FOnPointPointContact&& fOnPointPointContact,
        FOnPointLineSegmentContact&& fOnPointEdgeContact,
        FOnPointTriangleContact&& fOnPointTriangleContact);
    /**
     * @brief For each edge-face contact of half-edge `hei`, invoke the appropriate callback
     *
     * @tparam FOnPointLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex, 2>
     * eindsi, TIndex j)` for edge `eindsi` and point `j`
     * @tparam FOnLineSegmentLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex,
     * 2> eindsi, Eigen::Vector<TIndex, 2> eindsj)` for edge `eindsi` and edge `eindsj`
     * @param hei Half-edge index
     * @param fOnPointLineSegmentContact Point-edge contact handler
     * @param fOnLineSegmentLineSegmentContact Edge-edge contact handler
     */
    template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
    void ForEachHalfEdgeDynamicMeshContact(
        IndexType hei,
        FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
        FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact);
    /**
     * @brief For each edge-(static)face contact of half-edge `hei`, invoke the appropriate callback
     *
     * @tparam FOnPointLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex, 2>
     * eindsi, TIndex j)` for edge `eindsi` and point `j`
     * @tparam FOnLineSegmentLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex,
     * 2> eindsi, Eigen::Vector<TIndex, 2> eindsj)` for edge `eindsi` and edge `eindsj`
     * @param hei Half-edge index
     * @param fOnPointLineSegmentContact Point-edge contact handler
     * @param fOnLineSegmentLineSegmentContact Edge-edge contact handler
     */
    template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
    void ForEachHalfEdgeStaticMeshContact(
        IndexType hei,
        FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
        FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact);
    /**
     * @brief For each edge-face contact whose edge is incident on point `i`, invoke the appropriate
     * callback
     *
     * @tparam FOnPointLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex, 2>
     * eindsi, IndexType j)` for edge `eindsi` and point `j`
     * @tparam FOnLineSegmentLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex,
     * 2> eindsi, Eigen::Vector<TIndex, 2> eindsj)` for edges `eindsi` and `eindsj`
     * @param i Point index
     * @param fOnPointLineSegmentContact Point-edge contact handler
     * @param fOnLineSegmentLineSegmentContact Edge-edge contact handler
     * @note `eindsi(0) == i`
     */
    template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
    void ForEachHalfEdgeDynamicMeshContactIncidentOnPoint(
        IndexType i,
        FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
        FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact);
    /**
     * @brief For each edge-(static)face contact whose edge is incident on point `i`, invoke the
     * appropriate callback
     *
     * @tparam FOnPointLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex, 2>
     * eindsi, IndexType j)` for edge `eindsi` and point `j`
     * @tparam FOnLineSegmentLineSegmentContact Callable with signature `void(Eigen::Vector<TIndex,
     * 2> eindsi, Eigen::Vector<TIndex, 2> eindsj)` for edges `eindsi` and `eindsj`
     * @param i Point index
     * @param fOnPointLineSegmentContact Point-edge contact handler
     * @param fOnLineSegmentLineSegmentContact Edge-edge contact handler
     * @note `eindsi(0) == i`
     */
    template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
    void ForEachHalfEdgeStaticMeshContactIncidentOnPoint(
        IndexType i,
        FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
        FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact);
    /**
     * @brief For each triangle-(dynamic)point contact of triangle `fi`, invoke the appropriate
     * callback
     * @tparam FOnPointTriangleContact Callable with signature `void(TIndex i)` for point `i`
     * @param fi Triangle index
     * @param fOnPointTriangleContact Point-triangle contact handler
     */
    template <class FOnPointTriangleContact>
    void ForEachDynamicPointContactOnTriangle(
        IndexType fi,
        FOnPointTriangleContact&& fOnPointTriangleContact);
    /**
     * @brief For each triangle-(static)point contact of triangle `fi`, invoke the appropriate
     * callback
     * @tparam FOnPointTriangleContact Callable with signature `void(TIndex i)` for static point `i`
     * @param fi Triangle index
     * @param fOnPointTriangleContact Point-triangle contact handler
     */
    template <class FOnPointTriangleContact>
    void ForEachStaticPointContactOnTriangle(
        IndexType fi,
        FOnPointTriangleContact&& fOnPointTriangleContact);
    /**
     * @brief For each triangle-face contact whose triangle is incident on point `i`, invoke the
     * appropriate callback
     * @tparam FOnPointTriangleContact Callable with signature `void(Eigen::Vector<TIndex, 3> finds,
     * TIndex j)` for point `j`
     * @param i Point index
     * @param fOnPointTriangleContact Point-triangle contact handler
     */
    template <class FOnPointTriangleContact>
    void ForEachDynamicPointContactOnTrianglesIncidentOnPoint(
        IndexType i,
        FOnPointTriangleContact&& fOnPointTriangleContact);
    /**
     * @brief For each triangle-(static)point contact whose triangle is incident on static point
     * `i`, invoke the appropriate callback
     * @tparam FOnPointTriangleContact Callable with signature `void(Eigen::Vector<TIndex, 3> finds,
     * TIndex j)` for static point `j`
     * @param i Dynamic point index
     * @param fOnPointTriangleContact Point-triangle contact handler
     */
    template <class FOnPointTriangleContact>
    void ForEachStaticPointContactOnTrianglesIncidentOnPoint(
        IndexType i,
        FOnPointTriangleContact&& fOnPointTriangleContact);
    /**
     * @brief For each mesh-mesh contact, invoke the appropriate callback
     *
     * @tparam FOnVertexVertexContact Callable type with signature `void(TIndex i, TIndex j)` for
     * points `i` and `j`
     * @tparam FOnVertexEdgeContact Callable type with signature `void(TIndex i,
     * Eigen::Vector<TIndex, 2> einds)` for point `i` and edge `einds`
     * @tparam FOnVertexTriangleContact Callable type with signature `void(TIndex i,
     * Eigen::Vector<TIndex, 3> finds)` for point `i` and triangle `finds`
     * @tparam FOnEdgeEdgeContact Callable type with signature `void(Eigen::Vector<TIndex, 2>
     * eindsi, Eigen::Vector<TIndex, 2> eindsj)` for edges `eindsi` and `eindsj`
     * @param fOnVertexVertexContact Callback for vertex-vertex contacts
     * @param fOnVertexEdgeContact Callback for vertex-edge contacts
     * @param fOnVertexTriangleContact Callback for vertex-triangle contacts
     * @param fOnEdgeEdgeContact Callback for edge-edge contacts
     */
    template <
        class FOnVertexVertexContact,
        class FOnVertexEdgeContact,
        class FOnVertexTriangleContact,
        class FOnEdgeEdgeContact>
    void ForEachMeshMeshContact(
        FOnVertexVertexContact&& fOnVertexVertexContact,
        FOnVertexEdgeContact&& fOnVertexEdgeContact,
        FOnVertexTriangleContact&& fOnVertexTriangleContact,
        FOnEdgeEdgeContact&& fOnEdgeEdgeContact);
    /**
     * @brief For each mesh-environment contact, invoke the appropriate callback
     *
     * @tparam FOnVertexEnvironmentContact Callable type with signature `void(TIndex i)` for point
     * `i`
     * @tparam FOnEdgeEnvironmentContact Callable type with signature `void(Eigen::Vector<TIndex, 2>
     * einds)` for edge `einds`
     * @tparam FOnTriangleEnvironmentContact Callable type with signature
     * `void(Eigen::Vector<TIndex, 3> finds)` for triangle `finds`
     * @param fOnVertexEnvironmentContact Callback for vertex-environment contacts
     * @param fOnEdgeEnvironmentContact Callback for edge-environment contacts
     * @param fOnTriangleEnvironmentContact Callback for triangle-environment contacts
     */
    template <
        class FOnVertexEnvironmentVertexContact,
        class FOnVertexEnvironmentEdgeContact,
        class FOnVertexEnvironmentTriangleContact,
        class FOnEdgeEnvironmentVertexContact,
        class FOnEdgeEnvironmentEdgeContact,
        class FOnTriangleEnvironmentVertexContact>
    void ForEachMeshEnvironmentContact(
        FOnVertexEnvironmentVertexContact&& fOnVertexEnvironmentVertexContact,
        FOnVertexEnvironmentEdgeContact&& fOnVertexEnvironmentEdgeContact,
        FOnVertexEnvironmentTriangleContact&& fOnVertexEnvironmentTriangleContact,
        FOnEdgeEnvironmentVertexContact&& fOnEdgeEnvironmentVertexContact,
        FOnEdgeEnvironmentEdgeContact&& fOnEdgeEnvironmentEdgeContact,
        FOnTriangleEnvironmentVertexContact&& fOnTriangleEnvironmentVertexContact);
    /**
     * @brief Set the static geometry
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param meshes Mesh contact geometry representation
     */
    void SetStaticGeometry(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        MultiMesh<IndexType> meshes);
    /**
     * @brief Set the dynamic geometry
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param meshes Mesh contact geometry representation
     */
    void SetDynamicGeometry(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        MultiMesh<IndexType> meshes);
    /**
     * @brief Compute contact energies (potential, gradient, hessian) from current positions
     * @param x `3 x |# points|` or `3*|# points| x 1` point positions
     * @param xt `3 x |# points|` or `3*|# points| x 1` point positions s.t. `\Dot{x} = (x - xt) /
     * h`
     * @param h Time step size
     * @param computationFlags Flags indicating which quantities to compute
     */
    template <class TDerivedX, class TDerivedXt>
    void ComputeEnergies(
        Eigen::MatrixBase<TDerivedX> const& x,
        Eigen::MatrixBase<TDerivedXt> const& xt,
        ScalarType h,
        int computationFlags);
    /**
     * @brief For each mesh contact energy (i.e. mesh-mesh and mesh-env), invoke the callback
     * @tparam FOnMeshContactEnergy Callable type with signature
     * `template <int K> void(MeshContactEnergy<ScalarType, kStencil>& energy)` for
     * each mesh contact energy
     * @param fOnMeshContactEnergy Callback to invoke
     */
    template <class FOnMeshContactEnergy>
    void ForEachMeshContactEnergy(FOnMeshContactEnergy&& fOnMeshContactEnergy);
    /**
     * @brief Compute the total potential energy from all contacts
     * @return Total potential energy
     * @pre `ComputeEnergies()` has been called with the `Potential` flag
     */
    ScalarType Potential() const;
    /**
     * @brief Compute the total contact gradient
     * @return Total contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    auto Gradient() const -> Eigen::Vector<ScalarType, Eigen::Dynamic>;
    /**
     * @brief Compute the total contact gradient and add it to `g`
     * @tparam TDerivedg Writeable matrix type
     * @param g `3*|# points| x 1` or `3 x |# points|` total contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    template <class TDerivedg>
    void ToGradient(Eigen::MatrixBase<TDerivedg>& g) const;
    /**
     * @brief Compute the normal contact gradient
     * @return Normal contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    auto NormalGradient() const -> Eigen::Vector<ScalarType, Eigen::Dynamic>;
    /**
     * @brief Compute the normal contact gradient and add it to `g`
     * @tparam TDerivedg Writeable matrix type
     * @param g `3*|# points| x 1` or `3 x |# points|` normal contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    template <class TDerivedg>
    void ToNormalGradient(Eigen::MatrixBase<TDerivedg>& g) const;
    /**
     * @brief Compute the frictional contact gradient
     * @return Frictional contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    auto FrictionalGradient() const -> Eigen::Vector<ScalarType, Eigen::Dynamic>;
    /**
     * @brief Compute the frictional contact gradient and add it to `g`
     * @tparam TDerivedg Writeable matrix type
     * @param g `3*|# points| x 1` or `3 x |# points|` frictional contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    template <class TDerivedg>
    void ToFrictionalGradient(Eigen::MatrixBase<TDerivedg>& g) const;
    /**
     * @brief Get the total number of contacts
     */
    std::size_t NumContacts() const;
    /**
     * @brief Get the number of vertex-vertex contacts
     */
    std::size_t NumVertexVertexContacts() const;
    /**
     * @brief Get the number of vertex-edge contacts
     */
    std::size_t NumVertexEdgeContacts() const;
    /**
     * @brief Get the number of vertex-triangle contacts
     */
    std::size_t NumVertexTriangleContacts() const;
    /**
     * @brief Get the number of edge-edge contacts
     */
    std::size_t NumEdgeEdgeContacts() const;
    /**
     * @brief Get the Params object
     * @return Reference to the parameters
     */
    Params const& GetParams() const { return mParams; }
    /**
     * @brief Get the Params object
     * @return Reference to the parameters
     */
    Params& GetParams() { return mParams; }
    /**
     * @brief Get the static point positions
     * @return `3 x |# points|` static point positions
     */
    auto StaticPointPositions() const -> Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const&
    {
        return mXstatic;
    }
    /**
     * @brief Get the Dynamic Meshes object
     * @return MultiMesh<IndexType> const&
     */
    auto DynamicMeshes() const -> MultiMesh<IndexType> const& { return mDynamicMeshes; }
    /**
     * @brief Get the Static Meshes object
     * @return MultiMesh<IndexType> const&
     */
    auto StaticMeshes() const -> MultiMesh<IndexType> const& { return mStaticMeshes; }
    /**
     * @brief Get the Ogc Input object
     * @return ogc::Input<ScalarType, IndexType> const&
     */
    auto OgcInput() const -> ogc::Input<ScalarType, IndexType> const& { return mOgcInput; }
    /**
     * @brief Get the Ogc Input object
     * @return ogc::Input<ScalarType, IndexType>&
     */
    auto OgcInput() -> ogc::Input<ScalarType, IndexType>& { return mOgcInput; }
    /**
     * @brief Get the Ogc State object
     * @return ogc::State<ScalarType, IndexType> const&
     */
    auto OgcState() const -> ogc::State<ScalarType, IndexType> const& { return mOgcState; }
    /**
     * @brief Get the Ogc State object
     * @return ogc::State<ScalarType, IndexType>&
     */
    auto OgcState() -> ogc::State<ScalarType, IndexType>& { return mOgcState; }
    /**
     * @brief Get the point-point contact adjacency set
     * @return Reference to the point-point contact adjacency set
     */
    auto PointPointContacts() const { return mPointPointContacts; }
    /**
     * @brief Get the point-edge contact adjacency set
     * @return Reference to the point-edge contact adjacency set
     */
    auto PointEdgeContacts() const { return mPointEdgeContacts; }
    /**
     * @brief Get the point-triangle contact adjacency set
     * @return Reference to the point-triangle contact adjacency set
     */
    auto PointTriangleContacts() const { return mPointTriangleContacts; }
    /**
     * @brief Get the edge-edge contact adjacency set
     * @return Reference to the edge-edge contact adjacency set
     */
    auto EdgeEdgeContacts() const { return mEdgeEdgeContacts; }
    /**
     * @brief Reduce the thread-local contact sets from ogc::State into the contact adjacency sets
     *
     * This method merges all thread-local contact pairs from mOgcState into the
     * mPointPointContacts, mPointEdgeContacts, mPointTriangleContacts, and mEdgeEdgeContacts
     * adjacency sets.
     */
    void ReduceContactSets();
    /**
     * @brief Serialize to archive
     * @param archive Archive to serialize to
     */
    void Serialize(io::Archive& archive);
    /**
     * @brief Deserialize from archive
     * @param archive Archive to deserialize from
     */
    void Deserialize(io::Archive const& archive);

  private:
    Params mParams; ///< Mesh dynamics parameters

    /**
     * @brief Contact detection data structures and algorithms
     */
    Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> mXdynamic; ///< Dynamic point positions
    Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> mXstatic;  ///< Static point positions
    MultiMesh<IndexType> mDynamicMeshes;                    ///< Dynamic geometry
    MultiMesh<IndexType> mStaticMeshes;                     ///< Static geometry
    ogc::Input<ScalarType, IndexType> mOgcInput;            ///< OGC input data structures
    ogc::State<ScalarType, IndexType> mOgcState; ///< OGC transient algorithm data structures
    Eigen::Index mNumTruncatedPoints{0};         ///< Number of truncated points in last truncation
    bool mRequiresBoundsRecomputation{true};     ///< Whether bounds recomputation is required

    /**
     * @brief Contact constraint adjacency sets
     *
     * These store the contact pairs along with their associated constraint data.
     * The adjacency sets are populated by reducing the thread-local contact sets from ogc::State.
     */
    graph::DenseAdjacencySet<IndexType, MeshPointPointConstraint<ScalarType>, bool>
        mPointPointContacts; ///< Point-point contact pairs with constraint data
    graph::DenseAdjacencySet<IndexType, MeshPointEdgeConstraint<ScalarType>, bool>
        mPointEdgeContacts; ///< Point-edge contact pairs with constraint data
    graph::DenseAdjacencySet<IndexType, MeshPointTriangleConstraint<ScalarType>, bool>
        mPointTriangleContacts; ///< Point-triangle contact pairs with constraint data
    graph::DenseAdjacencySet<IndexType, MeshEdgeEdgeConstraint<ScalarType>, bool>
        mEdgeEdgeContacts; ///< Edge-edge contact pairs with constraint data
};

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithOgcParams(ogc::Params<TScalar> const& params)
{
    mOgcParams = params;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithFrictionalContact(TScalar mu, TScalar epsv)
{
    this->mu   = mu;
    this->epsv = epsv;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithNormalContact(TScalar kc)
{
    this->kc = kc;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::WithQueryRadiusInitialization(
    TScalar _rqstart,
    TScalar _betarq)
{
    this->rqstart = _rqstart;
    this->betarq  = _betarq;
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshDynamics<TScalar, TIndex>::Params&
MeshDynamics<TScalar, TIndex>::Params::Construct(bool bValidate)
{
    if (bValidate)
    {
        if (kc <= TScalar(0))
        {
            throw std::invalid_argument("MeshDynamics::Params::Construct(): kc must be positive.");
        }
        if (mu < TScalar(0))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): mu must be non-negative.");
        }
        if (betarq < TScalar(0))
        {
            throw std::invalid_argument(
                "MeshDynamics::Params::Construct(): betarq must be non-negative.");
        }
    }
    auto tau = TScalar(0.5) * mOgcParams.r;
    auto r   = mOgcParams.r;
    kcp      = tau * kc * (tau - r) * (tau - r);
    b        = (TScalar(0.5) * kc) * (r - tau) * (r - tau) + kcp * std::log(tau);
    return *this;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void
MeshDynamics<TScalar, TIndex>::Params::ComputeQueryRadius(TScalar inertialTargetDistance)
{
    mOgcParams.rq = std::max(mOgcParams.r, rqstart) + betarq * inertialTargetDistance;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::Activate(bool bActive)
{
    this->bDeactivate = not bActive;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::Deactivate()
{
    bDeactivate = true;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::Serialize(io::Archive& archive) const
{
    auto grp = archive.GetOrCreateGroup("pbat.sim.contact.MeshDynamics.Params");
    {
        auto paramsGrp = grp["mOgcParams"];
        mOgcParams.Serialize(paramsGrp);
    }
    grp.WriteMetaData("epsv", epsv);
    grp.WriteMetaData("kc", kc);
    grp.WriteMetaData("mu", mu);
    grp.WriteMetaData("rqstart", rqstart);
    grp.WriteMetaData("betarq", betarq);
    grp.WriteMetaData("kcp", kcp);
    grp.WriteMetaData("b", b);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Params::Deserialize(io::Archive const& archive)
{
    auto grp = archive["pbat.sim.contact.MeshDynamics.Params"];
    mOgcParams.Deserialize(grp["mOgcParams"]);
    if (grp.HasMetaData("epsv"))
        epsv = grp.ReadMetaData<TScalar>("epsv");
    if (grp.HasMetaData("kc"))
        kc = grp.ReadMetaData<TScalar>("kc");
    if (grp.HasMetaData("mu"))
        mu = grp.ReadMetaData<TScalar>("mu");
    if (grp.HasMetaData("rqstart"))
        rqstart = grp.ReadMetaData<TScalar>("rqstart");
    if (grp.HasMetaData("betarq"))
        betarq = grp.ReadMetaData<TScalar>("betarq");
    if (grp.HasMetaData("kcp"))
        kcp = grp.ReadMetaData<TScalar>("kcp");
    if (grp.HasMetaData("b"))
        b = grp.ReadMetaData<TScalar>("b");
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline MeshDynamics<TScalar, TIndex>::MeshDynamics(Params const& params) : mParams(params)
{
    mParams.Construct();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Construct(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xdynamic,
    MultiMesh<IndexType> dynamicMeshes,
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& Xstatic,
    MultiMesh<IndexType> staticMeshes)
{
    SetDynamicGeometry(Xdynamic, std::move(dynamicMeshes));
    SetStaticGeometry(Xstatic, std::move(staticMeshes));
    mOgcInput.Construct();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Initialize(geometry::Device device)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.Initialize");
    mOgcState.Initialize(device, mOgcInput, mParams.mOgcParams);
    mNumTruncatedPoints          = 0;
    mRequiresBoundsRecomputation = true;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedXkp1>
inline Eigen::Index
MeshDynamics<TScalar, TIndex>::TruncateDisplacedPositions(Eigen::MatrixBase<TDerivedXkp1>& Xkp1)
{
    auto mask = Eigen::Vector<bool, Eigen::Dynamic>::Constant(Xkp1.cols(), false);
    return TruncateDisplacedPositions(Xkp1, mask);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedXkp1, class TMask>
inline Eigen::Index MeshDynamics<TScalar, TIndex>::TruncateDisplacedPositions(
    Eigen::MatrixBase<TDerivedXkp1>& _Xkp1,
    Eigen::DenseBase<TMask> const& mask)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.TruncateDisplacedPositions");
    static_assert(
        std::is_convertible_v<typename TMask::Scalar, bool>,
        "Mask scalar type must be convertible to bool");
    Eigen::Index nTruncated{0};
    auto const nVertices = mDynamicMeshes.V.size();
    auto Xkp1            = _Xkp1.derived().reshaped(3, _Xkp1.size() / 3);
    tbb::parallel_for(Eigen::Index{0}, nVertices, [&](Eigen::Index v) {
        IndexType i          = mDynamicMeshes.V(v);
        bool const bIsMasked = static_cast<bool>(mask(i));
        if (bIsMasked)
            return;
        ScalarType const b             = mOgcState.bv(v);
        auto xk                        = mXdynamic.col(i);
        auto xkp1                      = Xkp1.col(i);
        Eigen::Vector<ScalarType, 3> d = xkp1 - xk;
        ScalarType const dnorm         = d.norm();
        bool const bIsWithinBound      = dnorm <= b;
        if (bIsWithinBound)
            return;
        // x^{k+1} = x^k + (d/|d|)*b = x^k + d * (b/|d|)
        d *= (b / dnorm);
        xkp1 = xk + d;
        common::AtomicAdd(nTruncated, Eigen::Index{1});
    });
    mNumTruncatedPoints += nTruncated;
    mRequiresBoundsRecomputation = mNumTruncatedPoints >= mParams.mOgcParams.gammae * nVertices;
    return nTruncated;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedDxkp1, class TMask>
inline Eigen::Index MeshDynamics<TScalar, TIndex>::TruncateDisplacements(
    Eigen::MatrixBase<TDerivedDxkp1>& _Dxkp1,
    Eigen::DenseBase<TMask> const& mask)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.TruncateDisplacements");
    static_assert(
        std::is_convertible_v<typename TMask::Scalar, bool>,
        "Mask scalar type must be convertible to bool");
    Eigen::Index nTruncated{0};
    auto const nVertices = mDynamicMeshes.V.size();
    auto Dxkp1           = _Dxkp1.derived().reshaped(3, _Dxkp1.size() / 3);
    tbb::parallel_for(Eigen::Index{0}, nVertices, [&](Eigen::Index v) {
        IndexType i          = mDynamicMeshes.V(v);
        bool const bIsMasked = static_cast<bool>(mask(i));
        if (bIsMasked)
            return;
        ScalarType const b        = mOgcState.bv(v);
        auto d                    = Dxkp1.col(i);
        ScalarType const dnorm    = d.norm();
        bool const bIsWithinBound = dnorm <= b;
        if (bIsWithinBound)
            return;
        // x^{k+1} = x^k + (d/|d|)*b = x^k + d * (b/|d|)
        d *= (b / dnorm);
        common::AtomicAdd(nTruncated, Eigen::Index{1});
    });
    mNumTruncatedPoints += nTruncated;
    mRequiresBoundsRecomputation = mNumTruncatedPoints >= mParams.mOgcParams.gammae * nVertices;
    return nTruncated;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::RequestDisplacementBoundsComputation()
{
    mRequiresBoundsRecomputation = true;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Index MeshDynamics<TScalar, TIndex>::NumTruncatedPoints() const
{
    return mNumTruncatedPoints;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline bool MeshDynamics<TScalar, TIndex>::RequiresBoundsComputation() const
{
    return mRequiresBoundsRecomputation;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedX>
inline void
MeshDynamics<TScalar, TIndex>::ComputeDisplacementBounds(Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.ComputeDisplacementBounds");
    if (mParams.bDeactivate)
    {
        mOgcState.bv.setConstant(std::numeric_limits<ScalarType>::max());
        mRequiresBoundsRecomputation = false;
        mNumTruncatedPoints          = 0;
        return;
    }
    mXdynamic = X.derived();
    mOgcState.PrepareForExecution(mOgcInput, mParams.mOgcParams);
    ogc::VertexFacetContactDetection(mOgcInput, mParams.mOgcParams, mOgcState);
    ogc::EdgeEdgeContactDetection(mOgcInput, mParams.mOgcParams, mOgcState);
    ogc::UpdateDisplacementBounds(mOgcInput, mParams.mOgcParams, mOgcState);
    mRequiresBoundsRecomputation = false;
    mNumTruncatedPoints          = 0;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void MeshDynamics<TScalar, TIndex>::SetStaticGeometry(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    MultiMesh<IndexType> meshes)
{
    mXstatic      = X;
    mStaticMeshes = std::move(meshes);
    mOgcInput.WithStaticGeometry(
        mXstatic,
        mStaticMeshes.E,
        mStaticMeshes.F,
        mStaticMeshes.GVHEp,
        mStaticMeshes.GVHEadj,
        mStaticMeshes.GHEF,
        mStaticMeshes.EHE);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
void MeshDynamics<TScalar, TIndex>::SetDynamicGeometry(
    Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
    MultiMesh<IndexType> meshes)
{
    mXdynamic      = X;
    mDynamicMeshes = std::move(meshes);
    mOgcInput.WithDynamicGeometry(
        mXdynamic,
        mDynamicMeshes.V,
        mDynamicMeshes.F,
        mDynamicMeshes.E,
        mDynamicMeshes.VP,
        mDynamicMeshes.FP,
        mDynamicMeshes.EP,
        mDynamicMeshes.GVHEp,
        mDynamicMeshes.GVHEadj,
        mDynamicMeshes.GHEF,
        mDynamicMeshes.EHE);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline TScalar MeshDynamics<TScalar, TIndex>::Potential() const
{
    ScalarType E{0};
    return E;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::Gradient() const
    -> Eigen::Vector<TScalar, Eigen::Dynamic>
{
    Eigen::Vector<TScalar, Eigen::Dynamic> grad(mXdynamic.size());
    grad.setZero();
    ToGradient(grad);
    return grad;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumContacts() const
{
    return NumVertexVertexContacts() + NumVertexEdgeContacts() + NumVertexTriangleContacts() +
           NumEdgeEdgeContacts();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumVertexVertexContacts() const
{
    return mPointPointContacts.Size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumVertexEdgeContacts() const
{
    return mPointEdgeContacts.Size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumVertexTriangleContacts() const
{
    return mPointTriangleContacts.Size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumEdgeEdgeContacts() const
{
    return mEdgeEdgeContacts.Size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::ReduceContactSets()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.ReduceContactSets");
    bool const bAssumeThreadLocalSetsAreDisjoint{false};
    bool const bAssumeGlobalSetIsDisjointFromThreadLocalSets{false};
    namespace mini = math::linalg::mini;
    using mini::FromEigen;
    mPointPointContacts.Reduce(
        mOgcState.mXX.begin(),
        mOgcState.mXX.end(),
        bAssumeThreadLocalSetsAreDisjoint,
        bAssumeGlobalSetIsDisjointFromThreadLocalSets,
        [](IndexType i, IndexType j) {
            MeshPointPointConstraint<ScalarType> C;
            // TODO: Initialize Lagrange multiplier lambda, slack variable s, and complementarity
            // relaxation mu
            return C;
        });
    mPointEdgeContacts.Reduce(
        mOgcState.mXE.begin(),
        mOgcState.mXE.end(),
        bAssumeThreadLocalSetsAreDisjoint,
        bAssumeGlobalSetIsDisjointFromThreadLocalSets,
        [](IndexType i, IndexType j) {
            MeshPointEdgeConstraint<ScalarType> C;
            // TODO: Initialize Lagrange multiplier lambda, slack variable s, and complementarity
            // relaxation mu
            return C;
        });
    mPointTriangleContacts.Reduce(
        mOgcState.mXF.begin(),
        mOgcState.mXF.end(),
        bAssumeThreadLocalSetsAreDisjoint,
        bAssumeGlobalSetIsDisjointFromThreadLocalSets,
        [](IndexType i, IndexType j) {
            MeshPointTriangleConstraint<ScalarType> C;
            // TODO: Initialize Lagrange multiplier lambda, slack variable s, and complementarity
            // relaxation mu
            return C;
        });
    mEdgeEdgeContacts.Reduce(
        mOgcState.mEE.begin(),
        mOgcState.mEE.end(),
        bAssumeThreadLocalSetsAreDisjoint,
        bAssumeGlobalSetIsDisjointFromThreadLocalSets,
        [](IndexType i, IndexType j) {
            MeshEdgeEdgeConstraint<ScalarType> C;
            // TODO: Initialize Lagrange multiplier lambda, slack variable s, and complementarity
            // relaxation mu
            return C;
        });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedg>
inline void
MeshDynamics<TScalar, TIndex>::ToGradient([[maybe_unused]] Eigen::MatrixBase<TDerivedg>& g) const
{
    // auto G                   = g.derived().reshaped(3, g.size() / 3);
    // auto fAccumulateGradient = [&](auto const& energies) {
    //     using math::linalg::mini::ToEigen;
    //     for (auto const& e : energies)
    //         G(Eigen::placeholders::all, e.stencil).reshaped() +=
    //             ToEigen(e.gradEn) + ToEigen(e.gradEf);
    // };
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::NormalGradient() const
    -> Eigen::Vector<TScalar, Eigen::Dynamic>
{
    Eigen::Vector<TScalar, Eigen::Dynamic> grad(mXdynamic.size());
    grad.setZero();
    ToNormalGradient(grad);
    return grad;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedg>
inline void MeshDynamics<TScalar, TIndex>::ToNormalGradient(Eigen::MatrixBase<TDerivedg>& g) const
{
    // auto G                   = g.derived().reshaped(3, g.size() / 3);
    // auto fAccumulateGradient = [&](auto const& energies) {
    //     using math::linalg::mini::ToEigen;
    //     for (auto const& e : energies)
    //         G(Eigen::placeholders::all, e.stencil).reshaped() += ToEigen(e.gradEn);
    // };
    // fAccumulateGradient(mVertexVertexEnergies);
    // fAccumulateGradient(mVertexEdgeEnergies);
    // fAccumulateGradient(mVertexTriangleEnergies);
    // fAccumulateGradient(mEdgeEdgeEnergies);
    // fAccumulateGradient(mVertexEnvironmentEnergies);
    // fAccumulateGradient(mEdgeEnvironmentEnergies);
    // fAccumulateGradient(mTriangleEnvironmentEnergies);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline auto MeshDynamics<TScalar, TIndex>::FrictionalGradient() const
    -> Eigen::Vector<TScalar, Eigen::Dynamic>
{
    Eigen::Vector<TScalar, Eigen::Dynamic> grad(mXdynamic.size());
    grad.setZero();
    ToFrictionalGradient(grad);
    return grad;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedg>
inline void
MeshDynamics<TScalar, TIndex>::ToFrictionalGradient(Eigen::MatrixBase<TDerivedg>& g) const
{
    // auto G                   = g.derived().reshaped(3, g.size() / 3);
    // auto fAccumulateGradient = [&](auto const& energies) {
    //     using math::linalg::mini::ToEigen;
    //     for (auto const& e : energies)
    //         G(Eigen::placeholders::all, e.stencil).reshaped() += ToEigen(e.gradEf);
    // };
    // fAccumulateGradient(mVertexVertexEnergies);
    // fAccumulateGradient(mVertexEdgeEnergies);
    // fAccumulateGradient(mVertexTriangleEnergies);
    // fAccumulateGradient(mEdgeEdgeEnergies);
    // fAccumulateGradient(mVertexEnvironmentEnergies);
    // fAccumulateGradient(mEdgeEnvironmentEnergies);
    // fAccumulateGradient(mTriangleEnvironmentEnergies);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Serialize(io::Archive& archive)
{
    auto grp = archive.GetOrCreateGroup("pbat.sim.contact.MeshDynamics");
    {
        auto paramsGrp = grp["mParams"];
        mParams.Serialize(paramsGrp);
    }
    grp.WriteData("mXdynamic", mXdynamic);
    {
        auto dynamicMeshesGrp = grp["mDynamicMeshes"];
        mDynamicMeshes.Serialize(dynamicMeshesGrp);
    }
    if (mXstatic.size() > 0)
    {
        grp.WriteData("mXstatic", mXstatic);
        {
            auto staticMeshesGrp = grp["mStaticMeshes"];
            mStaticMeshes.Serialize(staticMeshesGrp);
        }
    }
    grp.WriteMetaData("mNumTruncatedPoints", mNumTruncatedPoints);
    grp.WriteMetaData(
        "mRequiresBoundsRecomputation",
        static_cast<int>(mRequiresBoundsRecomputation));
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::Deserialize(io::Archive const& archive)
{
    auto grp = archive["pbat.sim.contact.MeshDynamics"];
    mParams.Deserialize(grp["mParams"]);
    mXdynamic =
        grp.ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("mXdynamic");
    mDynamicMeshes.Deserialize(grp["mDynamicMeshes"]);
    if (grp.HasMetaData("mNumTruncatedPoints"))
        mNumTruncatedPoints = grp.ReadMetaData<Eigen::Index>("mNumTruncatedPoints");
    if (grp.HasMetaData("mRequiresBoundsRecomputation"))
        mRequiresBoundsRecomputation =
            static_cast<bool>(grp.ReadMetaData<int>("mRequiresBoundsRecomputation"));
    mOgcInput.WithDynamicGeometry(
        mXdynamic,
        mDynamicMeshes.V,
        mDynamicMeshes.F,
        mDynamicMeshes.E,
        mDynamicMeshes.VP,
        mDynamicMeshes.FP,
        mDynamicMeshes.EP,
        mDynamicMeshes.GVHEp,
        mDynamicMeshes.GVHEadj,
        mDynamicMeshes.GHEF,
        mDynamicMeshes.EHE);
    if (grp.HasData("mXstatic") and grp.HasGroup("mStaticMeshes"))
    {
        mXstatic =
            grp.ReadData<Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>>("mXstatic");
        mStaticMeshes.Deserialize(grp["mStaticMeshes"]);
        mOgcInput.WithStaticGeometry(
            mXstatic,
            mStaticMeshes.E,
            mStaticMeshes.F,
            mStaticMeshes.GVHEp,
            mStaticMeshes.GVHEadj,
            mStaticMeshes.GHEF,
            mStaticMeshes.EHE);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <
    class FOnPointPointContact,
    class FOnPointLineSegmentContact,
    class FOnPointTriangleContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachPointDynamicMeshContact(
    IndexType i,
    FOnPointPointContact&& fOnPointPointContact,
    FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
    FOnPointTriangleContact&& fOnPointTriangleContact)
{
    auto vi = mDynamicMeshes.GXV(i);
    if (vi < 0)
        return;
    // mOgcState.ForEachDynamicContactFaceOfVertex(
    //     vi,
    //     [this, func = std::forward<FOnPointPointContact>(fOnPointPointContact)](IndexType j) {
    //         func(j);
    //     },
    //     [this, func = std::forward<FOnPointLineSegmentContact>(fOnPointLineSegmentContact)](
    //         IndexType he) {
    //         Eigen::Vector<IndexType, 2> const einds{
    //             geometry::IncomingVertex(mDynamicMeshes.F, he),
    //             geometry::OutgoingVertex(mDynamicMeshes.F, he)};
    //         func(einds);
    //     },
    //     [this, func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](IndexType
    //     f) {
    //         Eigen::Vector<IndexType, 3> const finds = mDynamicMeshes.F.col(f);
    //         func(finds);
    //     });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <
    class FOnPointPointContact,
    class FOnPointLineSegmentContact,
    class FOnPointTriangleContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachPointStaticMeshContact(
    IndexType i,
    FOnPointPointContact&& fOnPointPointContact,
    FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
    FOnPointTriangleContact&& fOnPointTriangleContact)
{
    auto vi = mDynamicMeshes.GXV(i);
    if (vi < 0)
        return;
    // mOgcState.ForEachStaticContactFaceOfVertex(
    //     vi,
    //     [this, func = std::forward<FOnPointPointContact>(fOnPointPointContact)](IndexType vj) {
    //         func(vj);
    //     },
    //     [this, func = std::forward<FOnPointLineSegmentContact>(fOnPointLineSegmentContact)](
    //         IndexType he) {
    //         Eigen::Vector<IndexType, 2> const einds{
    //             geometry::IncomingVertex(mStaticMeshes.F, he),
    //             geometry::OutgoingVertex(mStaticMeshes.F, he)};
    //         func(einds);
    //     },
    //     [this, func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](IndexType
    //     f) {
    //         Eigen::Vector<IndexType, 3> const finds = mStaticMeshes.F.col(f);
    //         func(finds);
    //     });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachHalfEdgeDynamicMeshContact(
    IndexType hei,
    FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
    FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact)
{
    Eigen::Vector<IndexType, 2> const eindsi{
        geometry::IncomingVertex(mDynamicMeshes.F, hei),
        geometry::OutgoingVertex(mDynamicMeshes.F, hei)};
    // mOgcState.ForEachDynamicContactFaceOfHalfEdge(
    //     hei,
    //     [this,
    //      &eindsi,
    //      func = std::forward<FOnPointLineSegmentContact>(fOnPointLineSegmentContact)](IndexType
    //      j) {
    //         func(eindsi, j);
    //     },
    //     [this,
    //      &eindsi,
    //      func =
    //      std::forward<FOnLineSegmentLineSegmentContact>(fOnLineSegmentLineSegmentContact)](
    //         IndexType hej) {
    //         Eigen::Vector<IndexType, 2> const eindsj{
    //             geometry::IncomingVertex(mDynamicMeshes.F, hej),
    //             geometry::OutgoingVertex(mDynamicMeshes.F, hej)};
    //         func(eindsi, eindsj);
    //     });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachHalfEdgeStaticMeshContact(
    IndexType hei,
    FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
    FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact)
{
    Eigen::Vector<IndexType, 2> const eindsi{
        geometry::IncomingVertex(mDynamicMeshes.F, hei),
        geometry::OutgoingVertex(mDynamicMeshes.F, hei)};
    // mOgcState.ForEachStaticContactFaceOfHalfEdge(
    //     hei,
    //     [&eindsi, func = std::forward<FOnPointLineSegmentContact>(fOnPointLineSegmentContact)](
    //         IndexType vj) { func(eindsi, vj); },
    //     [this,
    //      &eindsi,
    //      func =
    //      std::forward<FOnLineSegmentLineSegmentContact>(fOnLineSegmentLineSegmentContact)](
    //         IndexType hej) {
    //         Eigen::Vector<IndexType, 2> const eindsj{
    //             geometry::IncomingVertex(mStaticMeshes.F, hej),
    //             geometry::OutgoingVertex(mStaticMeshes.F, hej)};
    //         func(eindsi, eindsj);
    //     });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachHalfEdgeDynamicMeshContactIncidentOnPoint(
    IndexType i,
    FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
    FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact)
{
    auto hebegin = mDynamicMeshes.GVHEp(i);
    auto heend   = mDynamicMeshes.GVHEp(i + 1);
    for (IndexType hei : mDynamicMeshes.GVHEadj.segment(hebegin, heend - hebegin))
    {
        ForEachHalfEdgeDynamicMeshContact(
            hei,
            fOnPointLineSegmentContact,
            fOnLineSegmentLineSegmentContact);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachHalfEdgeStaticMeshContactIncidentOnPoint(
    IndexType i,
    FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
    FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact)
{
    auto hebegin = mDynamicMeshes.GVHEp(i);
    auto heend   = mDynamicMeshes.GVHEp(i + 1);
    for (IndexType hei : mDynamicMeshes.GVHEadj.segment(hebegin, heend - hebegin))
    {
        ForEachHalfEdgeStaticMeshContact(
            hei,
            fOnPointLineSegmentContact,
            fOnLineSegmentLineSegmentContact);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointTriangleContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachDynamicPointContactOnTriangle(
    IndexType fi,
    FOnPointTriangleContact&& fOnPointTriangleContact)
{
    // mOgcState.ForEachDynamicVertexContactOfTriangle(
    //     fi,
    //     [this, func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](IndexType
    //     i) {
    //         func(i);
    //     });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointTriangleContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachStaticPointContactOnTriangle(
    IndexType fi,
    FOnPointTriangleContact&& fOnPointTriangleContact)
{
    // mOgcState.ForEachStaticVertexContactOfTriangle(
    //     fi,
    //     [func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](IndexType vi) {
    //         func(vi);
    //     });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointTriangleContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachDynamicPointContactOnTrianglesIncidentOnPoint(
    IndexType i,
    FOnPointTriangleContact&& fOnPointTriangleContact)
{
    auto hebegin = mDynamicMeshes.GVHEp(i);
    auto heend   = mDynamicMeshes.GVHEp(i + 1);
    for (IndexType hei : mDynamicMeshes.GVHEadj.segment(hebegin, heend - hebegin))
    {
        IndexType fi                            = geometry::FaceOfHalfEdge(hei);
        Eigen::Vector<IndexType, 3> const finds = mDynamicMeshes.F.col(fi);
        ForEachDynamicPointContactOnTriangle(
            fi,
            [finds, func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](
                IndexType j) { func(finds, j); });
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointTriangleContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachStaticPointContactOnTrianglesIncidentOnPoint(
    IndexType i,
    FOnPointTriangleContact&& fOnPointTriangleContact)
{
    auto hebegin = mDynamicMeshes.GVHEp(i);
    auto heend   = mDynamicMeshes.GVHEp(i + 1);
    for (IndexType hei : mDynamicMeshes.GVHEadj.segment(hebegin, heend - hebegin))
    {
        IndexType fi                            = geometry::FaceOfHalfEdge(hei);
        Eigen::Vector<IndexType, 3> const finds = mDynamicMeshes.F.col(fi);
        ForEachStaticPointContactOnTriangle(
            fi,
            [finds, func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](
                IndexType j) { func(finds, j); });
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <
    class FOnVertexVertexContact,
    class FOnVertexEdgeContact,
    class FOnVertexTriangleContact,
    class FOnEdgeEdgeContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachMeshMeshContact(
    FOnVertexVertexContact&& fOnVertexVertexContact,
    FOnVertexEdgeContact&& fOnVertexEdgeContact,
    FOnVertexTriangleContact&& fOnVertexTriangleContact,
    FOnEdgeEdgeContact&& fOnEdgeEdgeContact)
{
    auto const nVerts = mDynamicMeshes.V.size();
    auto const nEdges = mDynamicMeshes.E.cols();
    auto const nFaces = mDynamicMeshes.F.cols();
    for (auto v = 0; v < nVerts; ++v)
    {
        auto const i = mDynamicMeshes.V(v);
        // mOgcState.ForEachDynamicContactFaceOfVertex(
        //     v,
        //     [&, func = std::forward<FOnVertexVertexContact>(fOnVertexVertexContact)](IndexType j)
        //     {
        //         func(i, j);
        //     },
        //     [&, func = std::forward<FOnVertexEdgeContact>(fOnVertexEdgeContact)](IndexType he) {
        //         Eigen::Vector<IndexType, 2> const einds{
        //             geometry::IncomingVertex(mDynamicMeshes.F, he),
        //             geometry::OutgoingVertex(mDynamicMeshes.F, he)};
        //         func(i, einds);
        //     },
        //     [&,
        //      func = std::forward<FOnVertexTriangleContact>(fOnVertexTriangleContact)](IndexType
        //      f) {
        //         Eigen::Vector<IndexType, 3> const finds = mDynamicMeshes.F.col(f);
        //         func(i, finds);
        //     });
    }
    for (auto e = 0; e < nEdges; ++e)
    {
        auto hei                                 = mDynamicMeshes.EHE(0, e);
        Eigen::Vector<IndexType, 2> const eindsi = mDynamicMeshes.E.col(e);
        // mOgcState.ForEachDynamicContactFaceOfHalfEdge(
        //     hei,
        //     [&]([[maybe_unused]] auto _) { /* no-op */ },
        //     [&, func = std::forward<FOnEdgeEdgeContact>(fOnEdgeEdgeContact)](IndexType hej) {
        //         Eigen::Vector<IndexType, 2> const eindsj{
        //             geometry::IncomingVertex(mDynamicMeshes.F, hej),
        //             geometry::OutgoingVertex(mDynamicMeshes.F, hej)};
        //         func(eindsi, eindsj);
        //     });
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <
    class FOnVertexEnvironmentVertexContact,
    class FOnVertexEnvironmentEdgeContact,
    class FOnVertexEnvironmentTriangleContact,
    class FOnEdgeEnvironmentVertexContact,
    class FOnEdgeEnvironmentEdgeContact,
    class FOnTriangleEnvironmentVertexContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachMeshEnvironmentContact(
    FOnVertexEnvironmentVertexContact&& fOnVertexEnvironmentVertexContact,
    FOnVertexEnvironmentEdgeContact&& fOnVertexEnvironmentEdgeContact,
    FOnVertexEnvironmentTriangleContact&& fOnVertexEnvironmentTriangleContact,
    FOnEdgeEnvironmentVertexContact&& fOnEdgeEnvironmentVertexContact,
    FOnEdgeEnvironmentEdgeContact&& fOnEdgeEnvironmentEdgeContact,
    FOnTriangleEnvironmentVertexContact&& fOnTriangleEnvironmentVertexContact)
{
    auto const nVerts = mDynamicMeshes.V.size();
    auto const nEdges = mDynamicMeshes.E.cols();
    auto const nFaces = mDynamicMeshes.F.cols();
    for (auto v = 0; v < nVerts; ++v)
    {
        IndexType const i = mDynamicMeshes.V(v);
        // mOgcState.ForEachStaticContactFaceOfVertex(
        //     v,
        //     [&,
        //      func = std::forward<FOnVertexEnvironmentVertexContact>(
        //          fOnVertexEnvironmentVertexContact)](IndexType j) { func(i, j); },
        //     [&,
        //      func =
        //      std::forward<FOnVertexEnvironmentEdgeContact>(fOnVertexEnvironmentEdgeContact)](
        //         IndexType hej) {
        //         Eigen::Vector<IndexType, 2> const einds{
        //             geometry::IncomingVertex(mStaticMeshes.F, hej),
        //             geometry::OutgoingVertex(mStaticMeshes.F, hej)};
        //         func(i, einds);
        //     },
        //     [&,
        //      func = std::forward<FOnVertexEnvironmentTriangleContact>(
        //          fOnVertexEnvironmentTriangleContact)](IndexType f) {
        //         Eigen::Vector<IndexType, 3> const finds = mStaticMeshes.F.col(f);
        //         func(i, finds);
        //     });
    }
    for (auto e = 0; e < nEdges; ++e)
    {
        auto hei                                 = mDynamicMeshes.EHE(0, e);
        Eigen::Vector<IndexType, 2> const eindsi = mDynamicMeshes.E.col(e);
        // mOgcState.ForEachStaticContactFaceOfHalfEdge(
        //     hei,
        //     [&,
        //      func =
        //      std::forward<FOnEdgeEnvironmentVertexContact>(fOnEdgeEnvironmentVertexContact)](
        //         IndexType j) { func(eindsi, j); },
        //     [&, func =
        //     std::forward<FOnEdgeEnvironmentEdgeContact>(fOnEdgeEnvironmentEdgeContact)](
        //         IndexType hej) {
        //         Eigen::Vector<IndexType, 2> const eindsj{
        //             geometry::IncomingVertex(mStaticMeshes.F, hej),
        //             geometry::OutgoingVertex(mStaticMeshes.F, hej)};
        //         func(eindsi, eindsj);
        //     });
    }
    for (auto f = 0; f < nFaces; ++f)
    {
        Eigen::Vector<IndexType, 3> const findsi = mDynamicMeshes.F.col(f);
        // mOgcState.ForEachStaticVertexContactOfTriangle(
        //     f,
        //     [&,
        //      func = std::forward<FOnTriangleEnvironmentVertexContact>(
        //          fOnTriangleEnvironmentVertexContact)](IndexType j) { func(findsi, j); });
    }
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHDYNAMICS_H
