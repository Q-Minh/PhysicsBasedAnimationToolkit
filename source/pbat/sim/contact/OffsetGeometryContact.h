/**
 * @file OffsetGeometryContact.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Header file for offset geometric contact model.
 * @version 0.1
 * @date 2025-11-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H
#define PBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H

#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"
#include "pbat/common/Concepts.h"
#include "pbat/geometry/Device.h"
#include "pbat/geometry/HalfEdges.h"
#include "pbat/io/Archive.h"

#include <array>
#include <embree4/rtcore.h>
#include <type_traits>
#include <utility>
#include <vector>

namespace pbat::sim::contact {

/**
 * @brief Parameters for Offset Geometry Contact (OGC).
 *
 * This bundles algorithm configuration knobs and capacities. Mesh data and device are
 * intentionally not stored here per design; this type is for configurable parameters only.
 */
struct OgcParams
{
    /**
     * @brief BVH build quality options.
     */
    enum class EBuildQuality {
        Low,    ///< Low build quality (fast build time)
        Medium, ///< Medium build quality (balanced)
        High    ///< High build quality (slow build time)
    };
    /**
     * @brief Scene construction features.
     */
    enum class ESceneFeatures {
        None,    ///< No special features
        Dynamic, ///< Dynamic scene
        Compact, ///< Compact representation
        Robust   ///< Robust representation
    };

    Scalar r{Scalar(0)};  ///< Contact radius
    Scalar rq{Scalar(0)}; ///< Query inflation radius

    ESceneFeatures eSceneFeatures{ESceneFeatures::None}; ///< Scene features
    EBuildQuality eSceneBvhQuality{EBuildQuality::Low};  ///< Scene BVH build quality
    EBuildQuality eMeshBvhQuality{EBuildQuality::Low};   ///< Mesh BVH build quality

    int nMaxVertexFaceContactsEstimate{
        16}; ///< Max vertex-face contacts estimate for memory pre-allocation
    int nMaxFaceVertexContactsEstimate{
        16}; ///< Max face-vertex contacts estimate for memory pre-allocation
    int nMaxEdgeFaceContactsEstimate{
        16}; ///< Max edge-face contacts estimate for memory pre-allocation

    Scalar gammap{0.45}; ///< Relaxation parameter for vertex displacement bound, must satisfy `0 <
                         ///< gammap < 0.5`

  public:
    /**
     * @brief Set contact and query radii.
     * @param _r Contact radius
     * @param _rq Query radius
     * @return Reference to this
     */
    PBAT_API OgcParams& WithRadii(Scalar _r, Scalar _rq);
    /**
     * @brief Set scene features.
     * @param features Scene features
     * @return Reference to this
     */
    PBAT_API OgcParams& WithSceneFeatures(ESceneFeatures features);
    /**
     * @brief Set both scene and mesh BVH build quality.
     * @param scene Build quality for scene
     * @param mesh Build quality for mesh
     * @return Reference to this
     */
    PBAT_API OgcParams& WithBuildQuality(EBuildQuality scene, EBuildQuality mesh);
    /**
     * @brief Set maximum number of contacts.
     * @param nvf Max vertex-face contacts estimate
     * @param nfv Max face-vertex contacts estimate
     * @param nef Max edge-face contacts estimate
     * @return Reference to this
     */
    PBAT_API OgcParams& WithMaxContactEstimates(int nvf, int nfv, int nef);
    /**
     * @brief Set displacement bound parameters.
     * @param gammap Relaxation parameter for vertex displacement bound, must satisfy `0 < gammap <
     * 0.5`
     * @return Reference to this
     */
    PBAT_API OgcParams& WithDisplacementBoundConfig(Scalar _gammap);
    /**
     * @brief Validate and construct the parameters.
     * @param bValidate Whether to validate parameters
     * @return Reference to this
     */
    PBAT_API OgcParams& Construct(bool bValidate = true);
    /**
     * @brief Serialize the parameters to an archive.
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the parameters from an archive.
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive const& archive);
};

namespace detail {

/**
 * @brief Private user data structure for the OGC internal functions.
 *
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct UserData;

/**
 * @brief Private user data structure for per-component OGC internal functions.
 *
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct UserDataByComponent;

} // namespace detail

/**
 * @brief API of Offset Geometric Contact (OGC) algorithm \cite chen_offset_2025 for multi-body
 * triangle mesh scene.
 *
 * This class does not own mesh topology or vertex positions. It only owns the acceleration
 * structures created within Embree. All API functions accept topology (triangles, edges) and
 * connected-component labels as parameters. Vertex positions must be supplied where needed.
 */
class OffsetGeometryContact
{
  public:
    using ScalarType = Scalar; ///< Type for vertex coordinates
    using IndexType  = Index;  ///< Type for indices into vertex arrays
    static_assert(std::is_signed_v<IndexType>, "IndexType must be a signed integer type");

    /**
     * @brief Default constructor
     */
    PBAT_API OffsetGeometryContact() = default;
    /**
     * @brief Construct an empty OGC object with given device.
     * @param device Spatial acceleration device
     */
    PBAT_API OffsetGeometryContact(geometry::Device device);
    /**
     * @brief Deleted copy constructor and copy assignment operator.
     */
    OffsetGeometryContact(OffsetGeometryContact const&)            = delete;
    OffsetGeometryContact& operator=(OffsetGeometryContact const&) = delete;
    /**
     * @brief Defaulted move constructor and move assignment operator.
     */
    OffsetGeometryContact(OffsetGeometryContact&&)            = default;
    OffsetGeometryContact& operator=(OffsetGeometryContact&&) = default;
    /**
     * @brief Construct and build the BVH scene from shared buffers.
     *
     * @param device Embree device wrapper
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into X)
     * @param E `2 x |# edges|` undirected edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param params OGC parameters
     */
    PBAT_API OffsetGeometryContact(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
        OgcParams const& params);
    /**
     * @brief Initialize OGC, i.e. build its spatial acceleration data structures.
     *
     * @param device Spatial acceleration device
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `|# vertices| x 1` vertex indices (global indices into X)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` undirected edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param params OGC parameters
     */
    PBAT_API void Initialize(
        geometry::Device device,
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
        OgcParams const& params);
    /**
     * @brief Prepare for contact iteration.
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param E `2 x |# edges|` undirected edge vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param EP `|# connected components| x 1` edge prefix
     * @param params OGC parameters
     */
    PBAT_API void PrepareIteration(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
        OgcParams const& params);
    /**
     * @brief Compute vertex-facet and face-facet contact sets.
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param GVHEp `|# points + 1|` point to half-edge prefix
     * @param GVHEadj `|# half edges|` point to half-edge adjacency
     * @param GHEF `2 x |# half edges|` half-edge to face adjacency
     * @param params OGC parameters
     */
    PBAT_API void VertexFacetContactDetection(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& GHEF,
        OgcParams const& params);
    /**
     * @brief Compute edge-facet contact sets.
     * @param device Spatial acceleration device
     * @param X `3 x |# points|` point positions (column-major: one point per column)
     * @param V `3 x |# vertices|` vertex positions (column-major: one vertex per column)
     * @param F `3 x |# triangles|` triangle vertex indices (global indices into V)
     * @param VP `|# connected components| x 1` vertex prefix
     * @param FP `|# connected components| x 1` face prefix
     * @param GVHEp `|# points + 1|` point to half-edge prefix
     * @param GVHEadj `|# half edges|` point to half-edge adjacency
     * @param EHE `2 x |# edges|` edge to half-edge adjacency
     * @param params OGC parameters
     */
    PBAT_API void EdgeEdgeContactDetection(
        Eigen::Ref<Eigen::Matrix<ScalarType, 3, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& E,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& EP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
        Eigen::Ref<Eigen::Matrix<IndexType, 2, Eigen::Dynamic> const> const& EHE,
        OgcParams const& params);
    /**
     * @brief Compute total vertex displacement bounds.
     * @pre `VertexFacetContactDetection` and `EdgeEdgeContactDetection` have been called.
     * @post `bv` is populated.
     */
    PBAT_API void ComputeDisplacementBounds(
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEp,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& GVHEadj,
        OgcParams const& params);
    /**
     * @brief Scene axis-aligned bounding box.
     */
    PBAT_API auto Bounds() const
        -> std::pair<Eigen::Vector<ScalarType, 3>, Eigen::Vector<ScalarType, 3>>;
    /**
     * @brief Serialize the OGC to an archive.
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize the OGC from an archive.
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive& archive);
    /**
     * @brief Destructor
     */
    PBAT_API ~OffsetGeometryContact();

  private:
    /**
     * @brief Destroy the underlying Embree scene.
     */
    void Destroy() noexcept;

  public:
    /**
     * @brief Contact face structure.
     */
    struct ContactFace
    {
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
        IndexType a;     ///< Face (vertex, edge or triangle) index
        IndexType eFace; ///< Face type indicator: (0 | 1 | 2) -> (triangle | edge | vertex)
    };

    // NOTE: We should try custom allocators on the contact sets to see if we can boost performance
    std::vector<std::vector<ContactFace>> FOGC; ///< `|# vertices|` per-vertex contact facet sets
    std::vector<std::vector<IndexType>> VOGC; ///< `|# triangles|` per-triangle contact vertex sets.
                                              ///< Stores vertex indices only.
    std::vector<std::vector<ContactFace>> EOGC; ///< `|# edges|` per-edge contact facet sets
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        bv; ///< `|# vertices|` array of vertex displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dminv; ///< `|# vertices|` array of vertex displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dminf; ///< `|# faces|` array of face displacement bounds
    Eigen::Vector<ScalarType, Eigen::Dynamic>
        dmine; ///< `|# half-edges|` array of half-edge displacement bounds

  private:
    RTCScene mVertexScene{nullptr}; ///< Opaque RTCScene
    RTCScene mEdgeScene{nullptr};   ///< Opaque RTCScene
    RTCScene mFaceScene{nullptr};   ///< Opaque RTCScene

    Eigen::Vector<bool, Eigen::Dynamic>
        mVertexLocks; ///< `|# verts|` array of locks for synchronized access
                      ///< to per-vertex contact facet sets.
    Eigen::Vector<bool, Eigen::Dynamic> mEdgeLocks; ///< `|# edges|` array of locks for synchronized
                                                    ///< access to per-edge contact facet sets.
    Eigen::Vector<bool, Eigen::Dynamic>
        mFacetLocks; ///< `|# triangles|` array of locks for synchronized
                     ///< access to per-triangle contact vertex sets.
    std::vector<detail::UserDataByComponent<ScalarType, IndexType>>
        mUserDataPerComponent; ///< Per-component user data for RTCBoundsFunction callbacks
};

/**
 * @brief Computes the triangle face (vertex, edge or triangle) nearest to a point's projection on
 * a triangle.
 * @param u First barycentric coordinate of the closest point on the triangle
 * @param v Second barycentric coordinate of the closest point on the triangle
 * @param w Third barycentric coordinate of the closest point on the triangle
 * @return The pair (a, eFace), where a is either a local vertex index or edge index, and eFace
 * indicates the type of face, i.e. (0 | 1 | 2) -> (triangle | edge | vertex)
 */
template <common::CFloatingPoint TScalar>
std::pair<int, int> ClosestFaceFacetToVertex(TScalar u, TScalar v, TScalar w);

/**
 * @brief Computes the
 *
 * @tparam TScalar
 * @param s Barycentric coordinate of closest point on edge 1
 * @param t Barycentric coordinate of closest point on edge 2
 * @param e1 Edge index of edge 1
 * @param e2 Edge index of edge 2
 * @param e1v Vertex indices of edge 1
 * @param e2v Vertex indices of edge 2
 * @return The tuple (a1, eFace1, a2, eFace2), where a1 and a2 are either vertex indices or
 * edge indices, and eFace1 and eFace2 indicate the type of face (vertex or edge), i.e. (0 | 1) ->
 * (edge | vertex)
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
std::tuple<int, int, int, int> ClosestFaceEdgeToEdge(
    TScalar s,
    TScalar t,
    TIndex e1,
    TIndex e2,
    std::array<TIndex, 2> e1v,
    std::array<TIndex, 2> e2v);

/**
 * @brief Determines if point x is in the vertex feasible region of vertex i.
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 * @param X `3 x |# points|` point positions
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param GVHEp `|# vertices + 1|` vertex to half-edge prefix
 * @param GVHEadj `|# half edges|` vertex to half-edge adjacency
 * @param x `3 x 1` query point
 * @param i Point (global) index corresponding to vertex
 * @return true if in vertex feasible region; false otherwise
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
bool IsVertexFeasible(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& GVHEp,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& GVHEadj,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex i);

/**
 * @brief Determines if point x is in the edge feasible region of half-edge he of face fi.
 * @tparam TScalar Scalar type
 * @tparam TIndex Index type
 * @param X `3 x |# points|` point positions
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param GHEF `2 x |# half edges|` half-edge to (adjacent face, opposite face)
 * @param x `3 x 1` query point
 * @param fi Face index of half-edge he
 * @param he Half-edge index
 * @return true if in edge feasible region; false otherwise
 */
template <common::CFloatingPoint TScalar, common::CIndex TIndex>
bool IsEdgeFeasible(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& GHEF,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex fi,
    TIndex he);

namespace detail {

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct UserData
{
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const* X;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* V;
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const* F;
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const* E;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* VP;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* FP;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* EP;
    std::vector<std::vector<OffsetGeometryContact::ContactFace>>* FOGC;
    std::vector<std::vector<TIndex>>* VOGC;
    std::vector<std::vector<OffsetGeometryContact::ContactFace>>* EOGC;
    Eigen::Vector<bool, Eigen::Dynamic>* mVertexLocks;
    Eigen::Vector<bool, Eigen::Dynamic>* mEdgeLocks;
    Eigen::Vector<bool, Eigen::Dynamic>* mFacetLocks;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dminv;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dminf;
    Eigen::Vector<TScalar, Eigen::Dynamic>* dmine;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* GVHEp;
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const* GVHEadj;
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const* GHEF;
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const* EHE;
    TScalar r;
    TScalar rq;
};

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
struct UserDataByComponent
{
    UserData<TScalar, TIndex>* userData;
    TIndex component;
};

} // namespace detail

template <common::CFloatingPoint TScalar>
std::pair<int, int> ClosestFaceFacetToVertex(TScalar u, TScalar v, TScalar w)
{
    int const nZeros     = (u == TScalar(0)) + (v == TScalar(0)) + (w == TScalar(0));
    bool const bIsVertex = (nZeros == 2);
    bool const bIsEdge   = (nZeros == 1);
    int eFace            = (bIsVertex * 2) + (bIsEdge * 1);
    int a                = bIsVertex * ((v == TScalar(1)) * 1 + (w == TScalar(1)) * 2) +
            bIsEdge * ((u == TScalar(0)) * 1 + (v == TScalar(0)) * 2);
    return {a, eFace};
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
std::tuple<int, int, int, int> ClosestFaceEdgeToEdge(
    TScalar s,
    TScalar t,
    TIndex e1,
    TIndex e2,
    std::array<TIndex, 2> e1v,
    std::array<TIndex, 2> e2v)
{
    // Vertex contact has s,t equal to 0 or 1
    bool bIsEdgeContact1 = s > TScalar(0) and s < TScalar(1);
    bool bIsEdgeContact2 = t > TScalar(0) and t < TScalar(1);
    // 0 -> edge, 1 -> vertex
    int eFace1 = (not bIsEdgeContact1) * 1;
    int eFace2 = (not bIsEdgeContact2) * 1;
    int a1     = bIsEdgeContact1 * e1 +
             (not bIsEdgeContact1) * ((s == TScalar(0)) * e1v[0] + (s == TScalar(1)) * e1v[1]);
    int a2 = bIsEdgeContact2 * e2 +
             (not bIsEdgeContact2) * ((t == TScalar(0)) * e2v[0] + (t == TScalar(1)) * e2v[1]);
    return {a1, eFace1, a2, eFace2};
}

/**
 * @brief Vectorize contact face index based on face type.
 *
 * Given a triangle mesh face `f`, a local index `alocal` (local vertex or local half-edge index),
 * and a face-type tag `eFace` where 0=triangle, 1=edge, 2=vertex, this computes the unified
 * contact index `a` as used by the OGC contact sets.
 *
 * Definition:
 * - Triangle-face:   a = f
 * - Edge-face:       a = 3*f + alocal  (half-edge index within face f)
 * - Vertex-face:     a = F(alocal, f)  (global vertex index)
 *
 * @tparam TIndex Index type
 * @tparam TDerivedF Derived Eigen type for face connectivity (3 x |#faces|)
 * @param F `3 x |# triangles|` triangle vertex indices
 * @param f Face index
 * @param alocal Local index within face f (0..2)
 * @param eFace Face type tag: 0=triangle, 1=edge, 2=vertex
 * @return Vectorized contact index `a`
 */
template <common::CIndex TIndex, class TDerivedF>
inline TIndex
VertexFacetContactFaceIndex(Eigen::DenseBase<TDerivedF> const& F, TIndex f, int alocal, int eFace)
{
    static_assert(
        TDerivedF::RowsAtCompileTime == 3,
        "F must have 3 rows representing triangle vertex indices.");
    return (eFace == 0) * f /* face contact, return face index */ +
           (eFace == 1) * (3 * f + alocal) /* edge contact, return half-edge index */
           + (eFace == 2) * F(alocal, f) /* vertex contact, return global point index */;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
bool IsVertexFeasible(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& GVHEp,
    Eigen::Ref<Eigen::Vector<TIndex, Eigen::Dynamic> const> const& GVHEadj,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex i)
{
    bool bInVertexFeasibleRegion{true};
    TIndex const hebegin               = GVHEp(i);
    TIndex const heend                 = GVHEp(i + 1);
    Eigen::Vector<TScalar, 3> const xv = X.col(i);
    for (TIndex he : GVHEadj(Eigen::seq(hebegin, heend - 1)))
    {
        TIndex const vp = geometry::OutgoingVertex(F, he);
        auto const xvp  = X.col(vp);
        bInVertexFeasibleRegion &= ((x - xv).dot(xv - xvp) >= TScalar(0));
    }
    return bInVertexFeasibleRegion;
}

template <common::CFloatingPoint TScalar, common::CIndex TIndex>
bool IsEdgeFeasible(
    Eigen::Ref<Eigen::Matrix<TScalar, 3, Eigen::Dynamic> const> const& X,
    Eigen::Ref<Eigen::Matrix<TIndex, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<TIndex, 2, Eigen::Dynamic> const> const& GHEF,
    Eigen::Vector<TScalar, 3> const& x,
    TIndex fi,
    TIndex he)
{
    TIndex fj = GHEF(1, he);
    TIndex i  = geometry::IncomingVertex(F, he);
    TIndex j  = geometry::OutgoingVertex(F, he);
    TIndex k  = geometry::OutgoingVertex(F, he, 1 /* step */);
    // Get the third vertex l of triangle fj that is not part of undirected edge (i,j)
    TIndex l = (F(0, fj) != i and F(0, fj) != j) * F(0, fj) +
               (F(1, fj) != i and F(1, fj) != j) * F(1, fj) +
               (F(2, fj) != i and F(2, fj) != j) * F(2, fj);
    Eigen::Vector<TScalar, 3> xi  = X.col(i);
    Eigen::Vector<TScalar, 3> xj  = X.col(j);
    Eigen::Vector<TScalar, 3> xk  = X.col(k);
    Eigen::Vector<TScalar, 3> xl  = X.col(l);
    Eigen::Vector<TScalar, 3> xij = xj - xi;
    TScalar xijn2                 = xij.squaredNorm();
    // Tangent to the plane spanned by triangle fi, perpendicular to edge (i,j)
    Eigen::Vector<TScalar, 3> pin = (xi - xk) + (xk - xi).dot(xij) / xijn2 * xij;
    // Tangent to the plane spanned by triangle fj, perpendicular to edge (i,j)
    Eigen::Vector<TScalar, 3> pjn = (xi - xl) + (xl - xi).dot(xij) / xijn2 * xij;
    bool bInEdgeFeasibleRegion =
        ((x - xi).dot(xj - xi) >= TScalar(0)) and // within half-plane of vertex i
        ((x - xj).dot(xi - xj) >= TScalar(0)) and // within half-plane of vertex j
        ((x - xi).dot(pin) >= TScalar(0)) and     // within half-plane perpendicular to fi
        ((x - xi).dot(pjn) >= TScalar(0));        // within half-plane perpendicular to fj
    return bInEdgeFeasibleRegion;
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_OFFSETGEOMETRYCONTACT_H
