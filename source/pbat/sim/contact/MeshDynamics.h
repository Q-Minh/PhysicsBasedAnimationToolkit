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

#include "MultiMesh.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Atomic.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/ClosestPointQueries.h"
#include "pbat/geometry/Device.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/io/Archive.h"
#include "pbat/profiling/Profiling.h"
#include "pbat/sim/contact/Friction.h"
#include "pbat/sim/contact/Potentials.h"
#include "pbat/sim/contact/ogc/Ogc.h"

#include <Eigen/Core>
#include <cmath>
#include <tbb/parallel_for.h>
#include <type_traits>
#include <vector>

namespace pbat::sim::contact {

/**
 * @brief Energy computation flags
 */
enum EMeshEnergyComputationFlags : int {
    Potential = 1 << 0, ///< Compute potential energy
    Gradient  = 1 << 1, ///< Compute contact gradients
    Hessian   = 1 << 2  ///< Compute contact hessians
};

/**
 * @brief Energy (derivatives)
 * @tparam kStencil Number of vertices involved in the contact
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex, int kStencil>
struct MeshContactEnergy
{
    static constexpr int kDofs = 3 * kStencil;   ///< Degrees of freedom involved
    TScalar En;                                  ///< Normal contact energy
    TScalar Ef;                                  ///< Frictional contact energy
    Eigen::Vector<TScalar, kDofs> gradEn;        ///< Normal contact energy gradient
    Eigen::Matrix<TScalar, kDofs, kDofs> hessEn; ///< Normal contact energy Hessian
    Eigen::Vector<TScalar, kDofs> gradEf;        ///< Frictional contact energy gradient
    Eigen::Matrix<TScalar, kDofs, kDofs> hessEf; ///< Frictional contact energy Hessian
    Eigen::Vector<TIndex, kStencil> stencil;     ///< Indices of involved vertices
};

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
        TScalar kc{1e8};    ///< OGC contact stiffness parameter, `kc > 0`
        TScalar mu{0.5};    ///< OGC friction coefficient, `mu >= 0`

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
         * @brief Construct the Params object
         * @param bValidate Whether to validate parameters
         * @return Reference to this
         */
        SelfType& Construct(bool bValidate = true);
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
     * @return Number of truncated points
     * @pre `Initialize()` has been called
     * @post Displacements have been truncated to satisfy the computed displacement bounds
     */
    template <class TDerivedXkp1>
    Eigen::Index TruncateDisplacement(Eigen::MatrixBase<TDerivedXkp1>& Xkp1);
    /**
     * @brief Truncate displacements to satisfy the computed displacement bounds
     * @tparam TDerivedXkp1 Writeable matrix type
     * @tparam TMask Eigen dense base s.t. TMask::Scalar is convertible to bool
     * @param Xkp1 `3 x |# points|` proposed new point positions (column-major: one point per
     * column)
     * @param mask `|# points| x 1` mask of points to ignore (true = ignore, false = process)
     * @return Number of truncated points
     * @pre `Initialize()` has been called
     * @post Displacements have been truncated to satisfy the computed displacement bounds
     */
    template <class TDerivedXkp1, class TMask>
    Eigen::Index TruncateDisplacement(
        Eigen::MatrixBase<TDerivedXkp1>& Xkp1,
        Eigen::DenseBase<TMask> const& mask);
    /**
     * @brief Get the number of truncated points from the last `TruncateDisplacement()`
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
     * @pre `TruncateDisplacement()` has been called
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
     * @param computationFlags Flags indicating which quantities to compute
     */
    template <class TDerivedX>
    void ComputeEnergies(Eigen::MatrixBase<TDerivedX> const& x, int computationFlags);
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
    Eigen::Vector<ScalarType, Eigen::Dynamic> Gradient() const;
    /**
     * @brief Compute the total contact gradient and add it to `g`
     * @tparam TDerivedg Writeable matrix type
     * @param g `3*|# points| x 1` or `3 x |# points|` total contact gradient
     * @pre `ComputeEnergies()` has been called with the `Gradient` flag
     */
    template <class TDerivedg>
    void ToGradient(Eigen::MatrixBase<TDerivedg>& g) const;
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
     * @brief Get the number of vertex-environment contacts
     */
    std::size_t NumVertexEnvironmentContacts() const;
    /**
     * @brief Get the number of edge-environment contacts
     */
    std::size_t NumEdgeEnvironmentContacts() const;
    /**
     * @brief Get the number of triangle-environment contacts
     */
    std::size_t NumTriangleEnvironmentContacts() const;
    /**
     * @brief Recomputes geometric quantities (triangle, half-edge, and vertex areas) from current
     * positions
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     */
    void UpdateGeometricQuantities(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X);
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
     * @brief Get the Ogc State object
     * @return ogc::State<ScalarType, IndexType> const&
     */
    auto OgcState() const -> ogc::State<ScalarType, IndexType> const& { return mOgcState; }

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

    std::vector<MeshContactEnergy<ScalarType, IndexType, 2>>
        mVertexVertexEnergies; ///< Vertex-vertex energies
    std::vector<MeshContactEnergy<ScalarType, IndexType, 3>>
        mVertexEdgeEnergies; ///< Vertex-edge energies
    std::vector<MeshContactEnergy<ScalarType, IndexType, 4>>
        mVertexTriangleEnergies; ///< Vertex-triangle energies
    std::vector<MeshContactEnergy<ScalarType, IndexType, 4>>
        mEdgeEdgeEnergies; ///< Edge-edge energies
    std::vector<MeshContactEnergy<ScalarType, IndexType, 1>>
        mVertexEnvironmentEnergies; ///< Vertex-environment energies
    std::vector<MeshContactEnergy<ScalarType, IndexType, 2>>
        mEdgeEnvironmentEnergies; ///< Edge-environment energies
    std::vector<MeshContactEnergy<ScalarType, IndexType, 3>>
        mTriangleEnvironmentEnergies; ///< Triangle-environment energies

    /**
     * @brief These geometric quantities are generally useful for contact dynamics
     */
    Eigen::Vector<ScalarType, Eigen::Dynamic> FA;  ///< `|# triangles| x 1` triangle areas
    Eigen::Vector<ScalarType, Eigen::Dynamic> HEA; ///< `|# half-edges| x 1` half-edge areas
    Eigen::Vector<ScalarType, Eigen::Dynamic> VA;  ///< `|# vertices| x 1` vertex areas

    /**
     * @brief Resize energy buffers for each contact type
     * @param nVertexVertexContacts Number of vertex-vertex dynamic contacts
     * @param nVertexEdgeContacts Number of vertex-edge dynamic contacts
     * @param nVertexTriangleContacts Number of vertex-triangle dynamic contacts
     * @param nEdgeEdgeContacts Number of edge-edge dynamic contacts
     * @param nVertexEnvironmentContacts Number of vertex-environment static contacts
     * @param nEdgeEnvironmentContacts Number of edge-environment static contacts
     * @param nTriangleEnvironmentContacts Number of triangle-environment static contacts
     */
    void ReserveContactEnergies(
        Eigen::Index nVertexVertexContacts,
        Eigen::Index nVertexEdgeContacts,
        Eigen::Index nVertexTriangleContacts,
        Eigen::Index nEdgeEdgeContacts,
        Eigen::Index nVertexEnvironmentContacts,
        Eigen::Index nEdgeEnvironmentContacts,
        Eigen::Index nTriangleEnvironmentContacts);
};

/**
 * @brief Compute vertex-vertex contact energy and its derivatives
 * @tparam TScalar Scalar type
 * @param xi `3 x 1` position of vertex i
 * @param xj `3 x 1` position of vertex j
 * @param r Contact radius
 * @param kc Normal contact stiffness
 * @param kcp
 * @param b
 * @param eFlags Energy computation flags
 * @return contact energy and its derivatives
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 2> VertexVertexContactEnergy(
    Eigen::Vector<TScalar, 3> const& xi,
    Eigen::Vector<TScalar, 3> const& xj,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags);

/**
 * @brief Compute vertex-edge contact energy and its derivatives
 * @tparam TScalar Scalar type
 * @param xi `3 x 1` position of vertex i
 * @param xa `3 x 1` position of edge endpoint a
 * @param xb `3 x 1` position of edge endpoint b
 * @param r Contact radius
 * @param kc Normal contact stiffness
 * @param kcp
 * @param b
 * @param eFlags Energy computation flags
 * @return contact energy and its derivatives
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 3> VertexEdgeContactEnergy(
    Eigen::Vector<TScalar, 3> const& xi,
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags);

/**
 * @brief Compute vertex-triangle contact energy and its derivatives
 * @tparam TScalar Scalar type
 * @param xi `3 x 1` position of vertex i
 * @param xa `3 x 1` position of triangle vertex a
 * @param xb `3 x 1` position of triangle vertex b
 * @param xc `3 x 1` position of triangle vertex c
 * @param r Contact radius
 * @param kc Normal contact stiffness
 * @param kcp
 * @param b
 * @param eFlags Energy computation flags
 * @return contact energy and its derivatives
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 4> VertexTriangleContactEnergy(
    Eigen::Vector<TScalar, 3> const& xi,
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    Eigen::Vector<TScalar, 3> const& xc,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags);

/**
 * @brief Compute edge-edge contact energy and its derivatives
 * @tparam TScalar Scalar type
 * @param xa `3 x 1` position of edge 1 endpoint a
 * @param xb `3 x 1` position of edge 1 endpoint b
 * @param xc `3 x 1` position of edge 2 endpoint c
 * @param xd `3 x 1` position of edge 2 endpoint d
 * @param r Contact radius
 * @param kc Normal contact stiffness
 * @param kcp
 * @param b
 * @param eFlags Energy computation flags
 * @return contact energy and its derivatives
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 4> EdgeEdgeContactEnergy(
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    Eigen::Vector<TScalar, 3> const& xc,
    Eigen::Vector<TScalar, 3> const& xd,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags);

/**
 * @brief Compute vertex-environment contact energy and its derivatives
 * @tparam TScalar Scalar type
 * @param xi `3 x 1` position of vertex i
 * @param xcp `3 x 1` closest point on environment
 * @param r Contact radius
 * @param kc Normal contact stiffness
 * @param kcp
 * @param b
 * @param eFlags Energy computation flags
 * @return contact energy and its derivatives
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 1> VertexEnvironmentContactEnergy(
    Eigen::Vector<TScalar, 3> const& xi,
    Eigen::Vector<TScalar, 3> const& xcp,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags);

/**
 * @brief Compute edge-environment contact energy and its derivatives
 * @tparam TScalar Scalar type
 * @param xa `3 x 1` position of edge endpoint a
 * @param xb `3 x 1` position of edge endpoint b
 * @param uv `2 x 1` barycentric coordinates of closest point on edge
 * @param xcp `3 x 1` closest point on environment
 * @param r Contact radius
 * @param kc Normal contact stiffness
 * @param kcp
 * @param b
 * @param eFlags Energy computation flags
 * @return contact energy and its derivatives
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 2> EdgeEnvironmentContactEnergy(
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    Eigen::Vector<TScalar, 2> const& uv,
    Eigen::Vector<TScalar, 3> const& xcp,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags);

/**
 * @brief Compute triangle-environment contact energy and its derivatives
 *
 * @tparam TScalar Scalar type
 * @param xa `3 x 1` position of triangle vertex a
 * @param xb `3 x 1` position of triangle vertex b
 * @param xc `3 x 1` position of triangle vertex c
 * @param uvw `3 x 1` barycentric coordinates of closest point on triangle
 * @param xcp `3 x 1` closest point on environment
 * @param r Contact radius
 * @param kc Normal contact stiffness
 * @param kcp
 * @param b
 * @param eFlags Energy computation flags
 * @return contact energy and its derivatives
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 3> TriangleEnvironmentContactEnergy(
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    Eigen::Vector<TScalar, 3> const& xc,
    Eigen::Vector<TScalar, 3> const& uvw,
    Eigen::Vector<TScalar, 3> const& xcp,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags);

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
    }
    auto tau = TScalar(0.5) * mOgcParams.r;
    auto r   = mOgcParams.r;
    kcp      = tau * kc * (tau - r) * (tau - r);
    b        = (TScalar(0.5) * kc) * (r - tau) * (r - tau) + kcp * std::log(tau);
    return *this;
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
MeshDynamics<TScalar, TIndex>::TruncateDisplacement(Eigen::MatrixBase<TDerivedXkp1>& Xkp1)
{
    auto mask = Eigen::Vector<bool, Eigen::Dynamic>::Constant(Xkp1.cols(), false);
    TruncateDisplacement(Xkp1, mask);
    return mNumTruncatedPoints;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedXkp1, class TMask>
inline Eigen::Index MeshDynamics<TScalar, TIndex>::TruncateDisplacement(
    Eigen::MatrixBase<TDerivedXkp1>& _Xkp1,
    Eigen::DenseBase<TMask> const& mask)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.TruncateDisplacementWithMask");
    static_assert(
        std::is_convertible_v<typename TMask::Scalar, bool>,
        "Mask scalar type must be convertible to bool");
    mNumTruncatedPoints  = 0;
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
        common::AtomicAdd(mNumTruncatedPoints, Eigen::Index{1});
    });
    mRequiresBoundsRecomputation = mNumTruncatedPoints >= mParams.mOgcParams.gammae * nVertices;
    return mNumTruncatedPoints;
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
    mXdynamic = X.derived();
    mOgcState.PrepareForExecution(mOgcInput, mParams.mOgcParams);
    ogc::VertexFacetContactDetection(mOgcInput, mParams.mOgcParams, mOgcState);
    ogc::EdgeEdgeContactDetection(mOgcInput, mParams.mOgcParams, mOgcState);
    ogc::UpdateDisplacementBounds(mOgcInput, mParams.mOgcParams, mOgcState);
    mRequiresBoundsRecomputation = false;
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
template <class TDerivedX>
inline void
MeshDynamics<TScalar, TIndex>::ComputeEnergies(Eigen::MatrixBase<TDerivedX> const& _x, int eFlags)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshDynamics.ComputeEnergies");
    Eigen::Index nVertexVertexContacts{0};
    Eigen::Index nVertexEdgeContacts{0};
    Eigen::Index nVertexTriangleContacts{0};
    Eigen::Index nEdgeEdgeContacts{0};
    Eigen::Index nVertexEnvironmentContacts{0};
    Eigen::Index nEdgeEnvironmentContacts{0};
    Eigen::Index nTriangleEnvironmentContacts{0};
    ForEachMeshMeshContact(
        [&](auto /*i*/, auto /*j*/) { ++nVertexVertexContacts; },
        [&](auto /*i*/, auto /*einds*/) { ++nVertexEdgeContacts; },
        [&](auto /*i*/, auto /*finds*/) { ++nVertexTriangleContacts; },
        [&](auto /*eindsi*/, auto /*eindsj*/) { ++nEdgeEdgeContacts; });
    ForEachMeshEnvironmentContact(
        [&](auto /*i*/, auto /*j*/) { ++nVertexEnvironmentContacts; },
        [&](auto /*i*/, auto /*einds*/) { ++nVertexEnvironmentContacts; },
        [&](auto /*i*/, auto /*finds*/) { ++nVertexEnvironmentContacts; },
        [&](auto /*eindsi*/, auto /*j*/) { ++nEdgeEnvironmentContacts; },
        [&](auto /*eindsi*/, auto /*eindsj*/) { ++nEdgeEnvironmentContacts; },
        [&](auto /*finds*/, auto /*j*/) { ++nTriangleEnvironmentContacts; });
    ReserveContactEnergies(
        nVertexVertexContacts,
        nVertexEdgeContacts,
        nVertexTriangleContacts,
        nEdgeEdgeContacts,
        nVertexEnvironmentContacts,
        nEdgeEnvironmentContacts,
        nTriangleEnvironmentContacts);
    mVertexVertexEnergies.clear();
    mVertexEdgeEnergies.clear();
    mVertexTriangleEnergies.clear();
    mEdgeEdgeEnergies.clear();
    mVertexEnvironmentEnergies.clear();
    mEdgeEnvironmentEnergies.clear();
    mTriangleEnvironmentEnergies.clear();
    // Compute energies
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::SMatrix;
    using math::linalg::mini::SVector;
    using math::linalg::mini::ToEigen;
    ScalarType r    = mParams.mOgcParams.r;
    ScalarType epsv = mParams.epsv;
    ScalarType mu   = mParams.mu;
    ScalarType kc   = mParams.kc;
    ScalarType kcp  = mParams.kcp;
    ScalarType b    = mParams.b;
    auto const& x   = _x.reshaped(3, _x.size() / 3);
    ForEachMeshMeshContact(
        [&](IndexType i, IndexType j) {
            Eigen::Vector<ScalarType, 3> const xi = x.col(i);
            Eigen::Vector<ScalarType, 3> const xj = x.col(j);
            auto E = VertexVertexContactEnergy<ScalarType, IndexType>(
                xi,
                xj,
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << i, j;
            mVertexVertexEnergies.push_back(std::move(E));
        },
        [&](IndexType i, Eigen::Vector<IndexType, 2> einds) {
            Eigen::Vector<ScalarType, 3> const xi = x.col(i);
            Eigen::Vector<ScalarType, 3> const xa = x.col(einds(0));
            Eigen::Vector<ScalarType, 3> const xb = x.col(einds(1));
            auto E                                = VertexEdgeContactEnergy<ScalarType, IndexType>(
                xi,
                xa,
                xb,
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << i, einds(0), einds(1);
            mVertexEdgeEnergies.push_back(std::move(E));
        },
        [&](IndexType i, Eigen::Vector<IndexType, 3> finds) {
            Eigen::Vector<ScalarType, 3> const xi = x.col(i);
            Eigen::Vector<ScalarType, 3> const xa = x.col(finds(0));
            Eigen::Vector<ScalarType, 3> const xb = x.col(finds(1));
            Eigen::Vector<ScalarType, 3> const xc = x.col(finds(2));
            auto E = VertexTriangleContactEnergy<ScalarType, IndexType>(
                xi,
                xa,
                xb,
                xc,
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << i, finds(0), finds(1), finds(2);
            mVertexTriangleEnergies.push_back(std::move(E));
        },
        [&](Eigen::Vector<IndexType, 2> eindsi, Eigen::Vector<IndexType, 2> eindsj) {
            Eigen::Vector<ScalarType, 3> const xa = x.col(eindsi(0));
            Eigen::Vector<ScalarType, 3> const xb = x.col(eindsi(1));
            Eigen::Vector<ScalarType, 3> const xc = x.col(eindsj(0));
            Eigen::Vector<ScalarType, 3> const xd = x.col(eindsj(1));
            auto E                                = EdgeEdgeContactEnergy<ScalarType, IndexType>(
                xa,
                xb,
                xc,
                xd,
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << eindsi(0), eindsi(1), eindsj(0), eindsj(1);
            mEdgeEdgeEnergies.push_back(std::move(E));
        });
    ForEachMeshEnvironmentContact(
        [&](IndexType i, IndexType j) {
            Eigen::Vector<ScalarType, 3> const xi = x.col(i);
            Eigen::Vector<ScalarType, 3> const yj = mXstatic.col(j);
            auto E = VertexEnvironmentContactEnergy<ScalarType, IndexType>(
                xi,
                yj,
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << i;
            mVertexEnvironmentEnergies.push_back(std::move(E));
        },
        [&](IndexType i, Eigen::Vector<IndexType, 2> eindsj) {
            Eigen::Vector<ScalarType, 3> const xi = x.col(i);
            Eigen::Vector<ScalarType, 3> const yc = mXstatic.col(eindsj(0));
            Eigen::Vector<ScalarType, 3> const yd = mXstatic.col(eindsj(1));
            SVector<ScalarType, 3> const xcp = geometry::ClosestPointQueries::PointOnLineSegment(
                FromEigen(xi),
                FromEigen(yc),
                FromEigen(yd));
            auto E = VertexEnvironmentContactEnergy<ScalarType, IndexType>(
                xi,
                ToEigen(xcp),
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << i;
            mVertexEnvironmentEnergies.push_back(std::move(E));
        },
        [&](IndexType i, Eigen::Vector<IndexType, 3> findsj) {
            Eigen::Vector<ScalarType, 3> const xi = x.col(i);
            Eigen::Vector<ScalarType, 3> const ya = mXstatic.col(findsj(0));
            Eigen::Vector<ScalarType, 3> const yb = mXstatic.col(findsj(1));
            Eigen::Vector<ScalarType, 3> const yc = mXstatic.col(findsj(2));
            SVector<ScalarType, 3> const xcp      = geometry::ClosestPointQueries::PointInTriangle(
                FromEigen(xi),
                FromEigen(ya),
                FromEigen(yb),
                FromEigen(yc));
            auto E = VertexEnvironmentContactEnergy<ScalarType, IndexType>(
                xi,
                ToEigen(xcp),
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << i;
            mVertexEnvironmentEnergies.push_back(std::move(E));
        },
        [&](Eigen::Vector<IndexType, 2> eindsi, IndexType j) {
            Eigen::Vector<ScalarType, 3> const xa = x.col(eindsi(0));
            Eigen::Vector<ScalarType, 3> const xb = x.col(eindsi(1));
            Eigen::Vector<ScalarType, 3> const yj = mXstatic.col(j);
            // Closest point on dynamic edge to static vertex
            SVector<ScalarType, 2> const uv = geometry::ClosestPointQueries::UvPointOnLineSegment(
                FromEigen(yj),
                FromEigen(xa),
                FromEigen(xb));
            auto E = EdgeEnvironmentContactEnergy<ScalarType, IndexType>(
                xa,
                xb,
                ToEigen(uv),
                yj,
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << eindsi(0), eindsi(1);
            mEdgeEnvironmentEnergies.push_back(std::move(E));
        },
        [&](Eigen::Vector<IndexType, 2> eindsi, Eigen::Vector<IndexType, 2> eindsj) {
            Eigen::Vector<ScalarType, 3> const xa = x.col(eindsi(0));
            Eigen::Vector<ScalarType, 3> const xb = x.col(eindsi(1));
            Eigen::Vector<ScalarType, 3> const yc = mXstatic.col(eindsj(0));
            Eigen::Vector<ScalarType, 3> const yd = mXstatic.col(eindsj(1));
            SVector<ScalarType, 2> const st       = geometry::ClosestPointQueries::LineSegments(
                FromEigen(xa),
                FromEigen(xb),
                FromEigen(yc),
                FromEigen(yd));
            SVector<ScalarType, 2> const u{1 - st(0), st(0)};
            Eigen::Vector<ScalarType, 3> const xcp = (1 - st(1)) * yc + st(1) * yd;
            auto E = EdgeEnvironmentContactEnergy<ScalarType, IndexType>(
                xa,
                xb,
                ToEigen(u),
                xcp,
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << eindsi(0), eindsi(1);
            mEdgeEnvironmentEnergies.push_back(std::move(E));
        },
        [&](Eigen::Vector<IndexType, 3> findsi, IndexType j) {
            Eigen::Vector<ScalarType, 3> const xa = x.col(findsi(0));
            Eigen::Vector<ScalarType, 3> const xb = x.col(findsi(1));
            Eigen::Vector<ScalarType, 3> const xc = x.col(findsi(2));
            Eigen::Vector<ScalarType, 3> const yj = mXstatic.col(j);
            SVector<ScalarType, 3> const uvw = geometry::ClosestPointQueries::UvwPointInTriangle(
                FromEigen(yj),
                FromEigen(xa),
                FromEigen(xb),
                FromEigen(xc));
            auto E = TriangleEnvironmentContactEnergy<ScalarType, IndexType>(
                xa,
                xb,
                xc,
                ToEigen(uvw),
                yj,
                r,
                kc,
                kcp,
                b,
                static_cast<EMeshEnergyComputationFlags>(eFlags));
            E.stencil << findsi(0), findsi(1), findsi(2);
            mTriangleEnvironmentEnergies.push_back(std::move(E));
        });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline TScalar MeshDynamics<TScalar, TIndex>::Potential() const
{
    ScalarType E{0};
    for (auto const& e : mVertexVertexEnergies)
        E += e.En + e.Ef;
    for (auto const& e : mVertexEdgeEnergies)
        E += e.En + e.Ef;
    for (auto const& e : mVertexTriangleEnergies)
        E += e.En + e.Ef;
    for (auto const& e : mEdgeEdgeEnergies)
        E += e.En + e.Ef;
    for (auto const& e : mVertexEnvironmentEnergies)
        E += e.En + e.Ef;
    for (auto const& e : mEdgeEnvironmentEnergies)
        E += e.En + e.Ef;
    for (auto const& e : mTriangleEnvironmentEnergies)
        E += e.En + e.Ef;
    return E;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline Eigen::Vector<TScalar, Eigen::Dynamic> MeshDynamics<TScalar, TIndex>::Gradient() const
{
    Eigen::Vector<TScalar, Eigen::Dynamic> grad(mXdynamic.size());
    grad.setZero();
    ToGradient(grad);
    return grad;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumVertexVertexContacts() const
{
    return mVertexVertexEnergies.size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumVertexEdgeContacts() const
{
    return mVertexEdgeEnergies.size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumVertexTriangleContacts() const
{
    return mVertexTriangleEnergies.size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumEdgeEdgeContacts() const
{
    return mEdgeEdgeEnergies.size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumVertexEnvironmentContacts() const
{
    return mVertexEnvironmentEnergies.size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumEdgeEnvironmentContacts() const
{
    return mEdgeEnvironmentEnergies.size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline std::size_t MeshDynamics<TScalar, TIndex>::NumTriangleEnvironmentContacts() const
{
    return mTriangleEnvironmentEnergies.size();
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class TDerivedg>
inline void MeshDynamics<TScalar, TIndex>::ToGradient(Eigen::MatrixBase<TDerivedg>& g) const
{
    auto G                   = g.derived().reshaped(3, g.size() / 3);
    auto fAccumulateGradient = [&](auto const& energies) {
        for (auto const& e : energies)
            G(Eigen::placeholders::all, e.stencil).reshaped() += e.gradEn + e.gradEf;
    };
    fAccumulateGradient(mVertexVertexEnergies);
    fAccumulateGradient(mVertexEdgeEnergies);
    fAccumulateGradient(mVertexTriangleEnergies);
    fAccumulateGradient(mEdgeEdgeEnergies);
    fAccumulateGradient(mVertexEnvironmentEnergies);
    fAccumulateGradient(mEdgeEnvironmentEnergies);
    fAccumulateGradient(mTriangleEnvironmentEnergies);
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
inline void MeshDynamics<TScalar, TIndex>::ReserveContactEnergies(
    Eigen::Index nVertexVertexContacts,
    Eigen::Index nVertexEdgeContacts,
    Eigen::Index nVertexTriangleContacts,
    Eigen::Index nEdgeEdgeContacts,
    Eigen::Index nVertexEnvironmentContacts,
    Eigen::Index nEdgeEnvironmentContacts,
    Eigen::Index nTriangleEnvironmentContacts)
{
    mVertexVertexEnergies.reserve(static_cast<std::size_t>(nVertexVertexContacts));
    mVertexEdgeEnergies.reserve(static_cast<std::size_t>(nVertexEdgeContacts));
    mVertexTriangleEnergies.reserve(static_cast<std::size_t>(nVertexTriangleContacts));
    mEdgeEdgeEnergies.reserve(static_cast<std::size_t>(nEdgeEdgeContacts));
    mVertexEnvironmentEnergies.reserve(static_cast<std::size_t>(nVertexEnvironmentContacts));
    mEdgeEnvironmentEnergies.reserve(static_cast<std::size_t>(nEdgeEnvironmentContacts));
    mTriangleEnvironmentEnergies.reserve(static_cast<std::size_t>(nTriangleEnvironmentContacts));
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
    mOgcState.ForEachDynamicContactFaceOfVertex(
        vi,
        [this, func = std::forward<FOnPointPointContact>(fOnPointPointContact)](IndexType j) {
            func(j);
        },
        [this, func = std::forward<FOnPointLineSegmentContact>(fOnPointLineSegmentContact)](
            IndexType he) {
            Eigen::Vector<IndexType, 2> einds{
                geometry::IncomingVertex(mDynamicMeshes.F, he),
                geometry::OutgoingVertex(mDynamicMeshes.F, he)};
            func(einds);
        },
        [this, func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](IndexType f) {
            Eigen::Vector<IndexType, 3> finds = mDynamicMeshes.F.col(f);
            func(finds);
        });
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
    mOgcState.ForEachStaticContactFaceOfVertex(
        vi,
        [this, func = std::forward<FOnPointPointContact>(fOnPointPointContact)](IndexType vj) {
            func(vj);
        },
        [this, func = std::forward<FOnPointLineSegmentContact>(fOnPointLineSegmentContact)](
            IndexType he) {
            Eigen::Vector<IndexType, 2> einds{
                geometry::IncomingVertex(mStaticMeshes.F, he),
                geometry::OutgoingVertex(mStaticMeshes.F, he)};
            func(einds);
        },
        [this, func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](IndexType f) {
            Eigen::Vector<IndexType, 3> finds = mStaticMeshes.F.col(f);
            func(finds);
        });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachHalfEdgeDynamicMeshContact(
    IndexType hei,
    FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
    FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact)
{
    Eigen::Vector<IndexType, 2> eindsi{
        geometry::IncomingVertex(mDynamicMeshes.F, hei),
        geometry::OutgoingVertex(mDynamicMeshes.F, hei)};
    mOgcState.ForEachDynamicContactFaceOfHalfEdge(
        hei,
        [this,
         &eindsi,
         func = std::forward<FOnPointLineSegmentContact>(fOnPointLineSegmentContact)](IndexType j) {
            func(eindsi, j);
        },
        [this,
         &eindsi,
         func = std::forward<FOnLineSegmentLineSegmentContact>(fOnLineSegmentLineSegmentContact)](
            IndexType hej) {
            Eigen::Vector<IndexType, 2> eindsj{
                geometry::IncomingVertex(mDynamicMeshes.F, hej),
                geometry::OutgoingVertex(mDynamicMeshes.F, hej)};
            func(eindsi, eindsj);
        });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointLineSegmentContact, class FOnLineSegmentLineSegmentContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachHalfEdgeStaticMeshContact(
    IndexType hei,
    FOnPointLineSegmentContact&& fOnPointLineSegmentContact,
    FOnLineSegmentLineSegmentContact&& fOnLineSegmentLineSegmentContact)
{
    Eigen::Vector<IndexType, 2> eindsi{
        geometry::IncomingVertex(mDynamicMeshes.F, hei),
        geometry::OutgoingVertex(mDynamicMeshes.F, hei)};
    mOgcState.ForEachStaticContactFaceOfHalfEdge(
        hei,
        [&eindsi, func = std::forward<FOnPointLineSegmentContact>(fOnPointLineSegmentContact)](
            IndexType vj) { func(eindsi, vj); },
        [this,
         &eindsi,
         func = std::forward<FOnLineSegmentLineSegmentContact>(fOnLineSegmentLineSegmentContact)](
            IndexType ej) {
            Eigen::Vector<IndexType, 2> eindsj = mStaticMeshes.E.col(ej);
            func(eindsi, eindsj);
        });
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
    mOgcState.ForEachDynamicVertexContactOfTriangle(
        fi,
        [this, func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](IndexType i) {
            func(i);
        });
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnPointTriangleContact>
inline void MeshDynamics<TScalar, TIndex>::ForEachStaticPointContactOnTriangle(
    IndexType fi,
    FOnPointTriangleContact&& fOnPointTriangleContact)
{
    mOgcState.ForEachStaticVertexContactOfTriangle(
        fi,
        [func = std::forward<FOnPointTriangleContact>(fOnPointTriangleContact)](IndexType vi) {
            func(vi);
        });
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
        IndexType fi                      = geometry::FaceOfHalfEdge(hei);
        Eigen::Vector<IndexType, 3> finds = mDynamicMeshes.F.col(fi);
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
        IndexType fi                      = geometry::FaceOfHalfEdge(hei);
        Eigen::Vector<IndexType, 3> finds = mDynamicMeshes.F.col(fi);
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
        mOgcState.ForEachDynamicContactFaceOfVertex(
            v,
            [&, func = std::forward<FOnVertexVertexContact>(fOnVertexVertexContact)](IndexType j) {
                func(i, j);
            },
            [&, func = std::forward<FOnVertexEdgeContact>(fOnVertexEdgeContact)](IndexType he) {
                Eigen::Vector<IndexType, 2> const einds{
                    geometry::IncomingVertex(mDynamicMeshes.F, he),
                    geometry::OutgoingVertex(mDynamicMeshes.F, he)};
                func(i, einds);
            },
            [&,
             func = std::forward<FOnVertexTriangleContact>(fOnVertexTriangleContact)](IndexType f) {
                Eigen::Vector<IndexType, 3> const finds = mDynamicMeshes.F.col(f);
                func(i, finds);
            });
    }
    for (auto e = 0; e < nEdges; ++e)
    {
        auto hei                                 = mDynamicMeshes.EHE(e, 0);
        Eigen::Vector<IndexType, 2> const eindsi = mDynamicMeshes.E.col(e);
        mOgcState.ForEachDynamicContactFaceOfHalfEdge(
            hei,
            [&]([[maybe_unused]] auto _) { /* no-op */ },
            [&, func = std::forward<FOnEdgeEdgeContact>(fOnEdgeEdgeContact)](IndexType hej) {
                Eigen::Vector<IndexType, 2> const eindsj{
                    geometry::IncomingVertex(mDynamicMeshes.F, hej),
                    geometry::OutgoingVertex(mDynamicMeshes.F, hej)};
                func(eindsi, eindsj);
            });
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
        mOgcState.ForEachStaticContactFaceOfVertex(
            v,
            [&,
             func = std::forward<FOnVertexEnvironmentVertexContact>(
                 fOnVertexEnvironmentVertexContact)](IndexType j) { func(i, j); },
            [&,
             func = std::forward<FOnVertexEnvironmentEdgeContact>(fOnVertexEnvironmentEdgeContact)](
                IndexType e) {
                Eigen::Vector<IndexType, 2> const einds = mStaticMeshes.E.col(e);
                func(i, einds);
            },
            [&,
             func = std::forward<FOnVertexEnvironmentTriangleContact>(
                 fOnVertexEnvironmentTriangleContact)](IndexType f) {
                Eigen::Vector<IndexType, 3> const finds = mStaticMeshes.F.col(f);
                func(i, finds);
            });
    }
    for (auto e = 0; e < nEdges; ++e)
    {
        auto hei                                 = mDynamicMeshes.EHE(e, 0);
        Eigen::Vector<IndexType, 2> const eindsi = mDynamicMeshes.E.col(e);
        mOgcState.ForEachStaticContactFaceOfHalfEdge(
            hei,
            [&,
             func = std::forward<FOnEdgeEnvironmentVertexContact>(fOnEdgeEnvironmentVertexContact)](
                IndexType j) { func(eindsi, j); },
            [&, func = std::forward<FOnEdgeEnvironmentEdgeContact>(fOnEdgeEnvironmentEdgeContact)](
                IndexType ej) {
                Eigen::Vector<IndexType, 2> const eindsj = mStaticMeshes.E.col(ej);
                func(eindsi, eindsj);
            });
    }
    for (auto f = 0; f < nFaces; ++f)
    {
        Eigen::Vector<IndexType, 3> const findsi = mDynamicMeshes.F.col(f);
        mOgcState.ForEachStaticVertexContactOfTriangle(
            f,
            [&,
             func = std::forward<FOnTriangleEnvironmentVertexContact>(
                 fOnTriangleEnvironmentVertexContact)](IndexType j) { func(findsi, j); });
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
template <class FOnMeshContactEnergy>
inline void
MeshDynamics<TScalar, TIndex>::ForEachMeshContactEnergy(FOnMeshContactEnergy&& fOnMeshContactEnergy)
{
    for (auto& E : mVertexVertexEnergies)
    {
        fOnMeshContactEnergy(E);
    }
    for (auto& E : mVertexEdgeEnergies)
    {
        fOnMeshContactEnergy(E);
    }
    for (auto& E : mVertexTriangleEnergies)
    {
        fOnMeshContactEnergy(E);
    }
    for (auto& E : mEdgeEdgeEnergies)
    {
        fOnMeshContactEnergy(E);
    }
    for (auto& E : mVertexEnvironmentEnergies)
    {
        fOnMeshContactEnergy(E);
    }
    for (auto& E : mEdgeEnvironmentEnergies)
    {
        fOnMeshContactEnergy(E);
    }
    for (auto& E : mTriangleEnvironmentEnergies)
    {
        fOnMeshContactEnergy(E);
    }
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 2> VertexVertexContactEnergy(
    Eigen::Vector<TScalar, 3> const& xi,
    Eigen::Vector<TScalar, 3> const& xj,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags)
{
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::SMatrix;
    using math::linalg::mini::SVector;
    using math::linalg::mini::ToEigen;
    MeshContactEnergy<TScalar, TIndex, 2> energy{};
    TScalar d = (xi - xj).norm();
    SVector<TScalar, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    if (eFlags | EMeshEnergyComputationFlags::Potential)
    {
        energy.En = dBdd(0);
        energy.Ef = TScalar(0); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Gradient)
    {
        energy.gradEn = ToEigen(
            contact::potentials::GradientWrtClosestPoints(
                FromEigen(xi),
                FromEigen(xj),
                d,
                dBdd(1)));
        energy.gradEf.setZero(); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Hessian)
    {
        energy.hessEn = ToEigen(
            contact::potentials::HessianWrtClosestPoints(
                FromEigen(xi),
                FromEigen(xj),
                d,
                dBdd(1),
                dBdd(2)));
        energy.hessEf.setZero(); // TODO
    }
    return energy;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 3> VertexEdgeContactEnergy(
    Eigen::Vector<TScalar, 3> const& xi,
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags)
{
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::Ones;
    using math::linalg::mini::SMatrix;
    using math::linalg::mini::SVector;
    using math::linalg::mini::ToEigen;
    MeshContactEnergy<TScalar, TIndex, 3> energy{};
    Ones<TScalar, 1, 1> w;
    SVector<TScalar, 2> const uv = geometry::ClosestPointQueries::UvPointOnLineSegment(
        FromEigen(xi),
        FromEigen(xa),
        FromEigen(xb));
    Eigen::Vector<TScalar, 3> const xcp = uv(0) * xa + uv(1) * xb;
    TScalar const d                     = (xi - xcp).norm();
    SVector<TScalar, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    if (eFlags | EMeshEnergyComputationFlags::Potential)
    {
        energy.En = dBdd(0);
        energy.Ef = TScalar(0); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Gradient)
    {
        energy.gradEn = ToEigen(
            contact::potentials::GradientWrtLinearlyInterpolatedClosestPoints(
                w,
                uv,
                FromEigen(xi),
                FromEigen(xcp),
                d,
                dBdd(1)));
        energy.gradEf.setZero(); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Hessian)
    {
        energy.hessEn = ToEigen(
            contact::potentials::HessianWrtLinearlyInterpolatedClosestPoints(
                w,
                uv,
                FromEigen(xi),
                FromEigen(xcp),
                d,
                dBdd(1),
                dBdd(2)));
        energy.hessEf.setZero(); // TODO
    }
    return energy;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 4> VertexTriangleContactEnergy(
    Eigen::Vector<TScalar, 3> const& xi,
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    Eigen::Vector<TScalar, 3> const& xc,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags)
{
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::Ones;
    using math::linalg::mini::SMatrix;
    using math::linalg::mini::SVector;
    using math::linalg::mini::ToEigen;
    MeshContactEnergy<TScalar, TIndex, 4> energy{};
    Ones<TScalar, 1, 1> w;
    SVector<TScalar, 3> const uvw = geometry::ClosestPointQueries::UvwPointInTriangle(
        FromEigen(xi),
        FromEigen(xa),
        FromEigen(xb),
        FromEigen(xc));
    Eigen::Vector<TScalar, 3> const xcp = uvw(0) * xa + uvw(1) * xb + uvw(2) * xc;
    TScalar const d                     = (xi - xcp).norm();
    SVector<TScalar, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    if (eFlags | EMeshEnergyComputationFlags::Potential)
    {
        energy.En = dBdd(0);
        energy.Ef = TScalar(0); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Gradient)
    {
        energy.gradEn = ToEigen(
            contact::potentials::GradientWrtLinearlyInterpolatedClosestPoints(
                w,
                uvw,
                FromEigen(xi),
                FromEigen(xcp),
                d,
                dBdd(1)));
        energy.gradEf.setZero(); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Hessian)
    {
        energy.hessEn = ToEigen(
            contact::potentials::HessianWrtLinearlyInterpolatedClosestPoints(
                w,
                uvw,
                FromEigen(xi),
                FromEigen(xcp),
                d,
                dBdd(1),
                dBdd(2)));
        energy.hessEf.setZero(); // TODO
    }
    return energy;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 4> EdgeEdgeContactEnergy(
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    Eigen::Vector<TScalar, 3> const& xc,
    Eigen::Vector<TScalar, 3> const& xd,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags)
{
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::SMatrix;
    using math::linalg::mini::SVector;
    using math::linalg::mini::ToEigen;
    MeshContactEnergy<TScalar, TIndex, 4> energy{};
    SVector<TScalar, 2> const st = geometry::ClosestPointQueries::LineSegments(
        FromEigen(xa),
        FromEigen(xb),
        FromEigen(xc),
        FromEigen(xd));
    SVector<TScalar, 2> const u{1 - st(0), st(0)};
    SVector<TScalar, 2> const v{1 - st(1), st(1)};
    Eigen::Vector<TScalar, 3> const xci = u(0) * xa + u(1) * xb;
    Eigen::Vector<TScalar, 3> const xcj = v(0) * xc + v(1) * xd;
    TScalar const d                     = (xci - xcj).norm();
    SVector<TScalar, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    if (eFlags | EMeshEnergyComputationFlags::Potential)
    {
        energy.En = dBdd(0);
        energy.Ef = TScalar(0); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Gradient)
    {
        energy.gradEn = ToEigen(
            contact::potentials::GradientWrtLinearlyInterpolatedClosestPoints(
                u,
                v,
                FromEigen(xci),
                FromEigen(xcj),
                d,
                dBdd(1)));
        energy.gradEf.setZero(); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Hessian)
    {
        energy.hessEn = ToEigen(
            contact::potentials::HessianWrtLinearlyInterpolatedClosestPoints(
                u,
                v,
                FromEigen(xci),
                FromEigen(xcj),
                d,
                dBdd(1),
                dBdd(2)));
        energy.hessEf.setZero(); // TODO
    }
    return energy;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 1> VertexEnvironmentContactEnergy(
    Eigen::Vector<TScalar, 3> const& xi,
    Eigen::Vector<TScalar, 3> const& xcp,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags)
{
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::SMatrix;
    using math::linalg::mini::SVector;
    using math::linalg::mini::ToEigen;
    MeshContactEnergy<TScalar, TIndex, 1> energy{};
    TScalar const d = (xi - xcp).norm();
    SVector<TScalar, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    if (eFlags | EMeshEnergyComputationFlags::Potential)
    {
        energy.En = dBdd(0);
        energy.Ef = TScalar(0); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Gradient)
    {
        energy.gradEn = ToEigen(
            contact::potentials::GradientSegmentWrtClosestPoints(
                FromEigen(xi),
                FromEigen(xcp),
                d,
                dBdd(1),
                0));
        energy.gradEf.setZero(); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Hessian)
    {
        energy.hessEn = ToEigen(
            contact::potentials::HessianBlockWrtClosestPoints(
                FromEigen(xi),
                FromEigen(xcp),
                d,
                dBdd(1),
                dBdd(2),
                0,
                0));
        energy.hessEf.setZero(); // TODO
    }
    return energy;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 2> EdgeEnvironmentContactEnergy(
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    Eigen::Vector<TScalar, 2> const& uv,
    Eigen::Vector<TScalar, 3> const& xcp,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags)
{
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::SMatrix;
    using math::linalg::mini::SVector;
    using math::linalg::mini::ToEigen;
    MeshContactEnergy<TScalar, TIndex, 2> energy{};
    Eigen::Vector<TScalar, 3> const xci = uv(0) * xa + uv(1) * xb;
    TScalar const d                     = (xci - xcp).norm();
    SVector<TScalar, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    if (eFlags | EMeshEnergyComputationFlags::Potential)
    {
        energy.En = dBdd(0);
        energy.Ef = TScalar(0); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Gradient)
    {
        energy.gradEn = ToEigen(
            contact::potentials::GradientWrtLinearlyInterpolatedClosestPoints(
                FromEigen(uv),
                FromEigen(xci),
                FromEigen(xcp),
                d,
                dBdd(1)));
        energy.gradEf.setZero(); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Hessian)
    {
        energy.hessEn = ToEigen(
            contact::potentials::HessianWrtLinearlyInterpolatedClosestPoints(
                FromEigen(uv),
                FromEigen(xci),
                FromEigen(xcp),
                d,
                dBdd(1),
                dBdd(2)));
        energy.hessEf.setZero(); // TODO
    }
    return energy;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
MeshContactEnergy<TScalar, TIndex, 3> TriangleEnvironmentContactEnergy(
    Eigen::Vector<TScalar, 3> const& xa,
    Eigen::Vector<TScalar, 3> const& xb,
    Eigen::Vector<TScalar, 3> const& xc,
    Eigen::Vector<TScalar, 3> const& uvw,
    Eigen::Vector<TScalar, 3> const& xcp,
    TScalar r,
    TScalar kc,
    TScalar kcp,
    TScalar b,
    EMeshEnergyComputationFlags eFlags)
{
    using math::linalg::mini::FromEigen;
    using math::linalg::mini::SMatrix;
    using math::linalg::mini::SVector;
    using math::linalg::mini::ToEigen;
    MeshContactEnergy<TScalar, TIndex, 3> energy{};
    Eigen::Vector<TScalar, 3> const xci = uvw(0) * xa + uvw(1) * xb + uvw(2) * xc;
    TScalar const d                     = (xci - xcp).norm();
    SVector<TScalar, 3> dBdd =
        contact::potentials::QuadraticToLogBarrierTwoStageActivation<2>(d, r, kc, kcp, b);
    if (eFlags | EMeshEnergyComputationFlags::Potential)
    {
        energy.En = dBdd(0);
        energy.Ef = TScalar(0); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Gradient)
    {
        energy.gradEn = ToEigen(
            contact::potentials::GradientWrtLinearlyInterpolatedClosestPoints(
                FromEigen(uvw),
                FromEigen(xci),
                FromEigen(xcp),
                d,
                dBdd(1)));
        energy.gradEf.setZero(); // TODO
    }
    if (eFlags | EMeshEnergyComputationFlags::Hessian)
    {
        energy.hessEn = ToEigen(
            contact::potentials::HessianWrtLinearlyInterpolatedClosestPoints(
                FromEigen(uvw),
                FromEigen(xci),
                FromEigen(xcp),
                d,
                dBdd(1),
                dBdd(2)));
        energy.hessEf.setZero(); // TODO
    }
    return energy;
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHDYNAMICS_H
