/**
 * @file MultibodyMeshMixedCcdDcd.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains a multibody (triangle) mesh continuous collision detection system.
 * @date 2025-03-25
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_CONTACT_MULTIBODYMESHMIXEDCCDDCD_H
#define PBAT_SIM_CONTACT_MULTIBODYMESHMIXEDCCDDCD_H

#include "pbat/Aliases.h"
#include "pbat/common/CountingSort.h"
#include "pbat/geometry/AabbKdTreeHierarchy.h"
#include "pbat/geometry/AabbRadixTreeHierarchy.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/geometry/EdgeEdgeCcd.h"
#include "pbat/geometry/IntersectionQueries.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/geometry/PointTriangleCcd.h"
#include "pbat/math/linalg/mini/Eigen.h"
#include "pbat/profiling/Profiling.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace pbat::sim::contact {

/**
 * @brief Multibody triangle mesh continuous collision detection system
 *
 * This class encapsulates an algorithm that maintains a pseudo active set of mesh-mesh
 * non-interpenetration constraints. The algorithm is based on a combination of discrete collision
 * detection (DCD) and continuous collision detection (CCD) techniques. The algorithm is designed
 * to be used in a multibody simulation context where the mesh objects are part of a larger
 * multibody system.
 *
 * We consider potentially colliding vertex-triangle and edge-edge pairs as contact constraints.
 * We maintain a cache of pairs (v,o) and (e,o) of active vertices and edges penetrating object o
 * between time steps to trim down DCD to only those vertices and edges that are already in contact.
 * Every other inactive vertex and edge is checked for earliest intersection using CCD in the linear
 * trajectory
 * \f$ \mathbf{x} = (1-\Delta t) \mathbf{x}(t) + \Delta t \mathbf{x}(t+1) \; \forall \; \Delta t \in
 * [0,1]\f$.`
 *
 * We adopt a tree of tree approach for broadphase culling. In both DCD and CCD, we first perform a
 * body-level broadphase using a BVH over all bodies in the world. For each potentially colliding
 * body pair (bi,bj), we perform a mesh-level broadphase using BVHs over the mesh vertices, edges
 * and triangles of bi and bj. For DCD, we consider vertex-triangle and edge-triangle pairs. For
 * CCD, we consider vertex-triangle and edge-edge pairs. After active set determination, we finalize
 * the active set by removing resolved contact pairs and marking active vertices/edges as inactive
 * if they do not intersect any tetrahedron (i.e. do not penetrate the bodies' volume).
 *
 * At a high level, our algorithm can be summarized as follows.
 *
 * A) The active set determination:
 * 1. Report DCD vertex-triangle and edge-triangle contacts for active vertices and edges.
 * 2. Report CCD vertex-triangle and edge-edge contacts for inactive vertices and edges.
 * 3. Mark each (v,o) and (e,o) pair of active vertices and edges potentially penetrating objects o
 * (in steps 1. and 2.).
 *
 * B) The active set finalization:
 * 1. Deactivate any active vertex/edge if they do not intersect a tetrahedron. If an edge does
 * intersect a tetrahedron, we also need to determine a point on the edge that penetrates the
 * tetrahedron, which will be use to create an edge-triangle contact in the next time step.
 * 2. Stream-compact remaining active vertices/edges and penetrated bodies.
 *
 * @note We should parallelize our algorithm via multi-threading and SIMD instructions for massive
 * speedup. Using 32-bit floating point numbers and integer indices would also speed up the
 * computation by leveraging at most twice the SIMD throughput and increasing cache hits.
 *
 */
class MultibodyMeshMixedCcdDcd
{
  public:
    static auto constexpr kDims = 3; ///< Number of spatial dimensions
    using BodyBvh               = geometry::AabbRadixTreeHierarchy<kDims>; ///< BVH over bodies
    using VertexBvh             = geometry::AabbKdTreeHierarchy<kDims>; ///< BVH over mesh vertices
    using EdgeBvh               = geometry::AabbKdTreeHierarchy<kDims>; ///< BVH over mesh edges
    using TriangleBvh           = geometry::AabbKdTreeHierarchy<kDims>; ///< BVH over mesh triangles
    using TetrahedronBvh = geometry::AabbKdTreeHierarchy<kDims>; ///< BVH over mesh tetrahedra

    /**
     * @brief Vertex-triangle contact pair (i,f)
     */
    struct VertexTriangleContact
    {
        using TriangleVertices = math::linalg::mini::SVector<Index, 3>;

        Index i, f;           ///< Vertex-triangle pair
        TriangleVertices fjs; ///< Triangle vertices
    };
    /**
     * @brief Edge-edge contact pair
     */
    struct EdgeEdgeContact
    {
        using EdgeBarycentricCoordinates = math::linalg::mini::SVector<Scalar, 2>;
        using EdgeVertices               = math::linalg::mini::SVector<Index, 2>;

        Index ei, ej;          ///< Edge-edge pair
        EdgeVertices eis, ejs; ///< Edge indices
        EdgeBarycentricCoordinates
            beta; ///< Barycentric weights of the intersection point on each edges
    };

    /**
     * @brief Construct a new TriangleMeshMultibodyCcd object from input triangle meshes
     *
     * @tparam TDerivedX Eigen type of the input vertex positions
     * @tparam TDerivedVP Eigen type of the input vertex prefix sum
     * @tparam TDerivedEP Eigen type of the input edge prefix sum
     * @tparam TDerivedFP Eigen type of the input face prefix sum
     * @tparam TDerivedTP Eigen type of the input tetrahedron prefix sum
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param VP `|# objects + 1| x 1` prefix sum of vertex pointers into `V` s.t.
     * `V(VP(o):VP(o+1))` are collision vertices of object `o`
     * @param V `|# vertices|` vertex array
     * @param EP `|# objects + 1| x 1` prefix sum of edge pointers into `E` s.t. `E(EP(o):EP(o+1))`
     * are collision edges of object `o`
     * @param E `2 x |# edges|` edge array
     * @param FP `|# objects + 1| x 1` prefix sum of triangle pointers into `F` s.t.
     * `F(FP(o):FP(o+1))` are collision triangles of object `o`
     * @param F `3 x |# triangles|` triangle array
     * @param TP `|# objects + 1| x 1` prefix sum of tetrahedron pointers into `T` s.t.
     * `T(TP(o):TP(o+1))` are collision tetrahedra of object `o`
     * @param T `4 x |# tetrahedra|` tetrahedron array
     */
    template <
        class TDerivedX,
        class TDerivedVP,
        class TDerivedEP,
        class TDerivedFP,
        class TDerivedTP>
    MultibodyMeshMixedCcdDcd(
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::DenseBase<TDerivedVP> const& VP,
        Eigen::Ref<IndexVectorX const> const& V,
        Eigen::DenseBase<TDerivedEP> const& EP,
        Eigen::Ref<IndexMatrix<2, Eigen::Dynamic> const> const& E,
        Eigen::DenseBase<TDerivedFP> const& FP,
        Eigen::Ref<IndexMatrix<3, Eigen::Dynamic> const> const& F,
        Eigen::DenseBase<TDerivedTP> const& TP,
        Eigen::Ref<IndexMatrix<4, Eigen::Dynamic> const> const& T);
    /**
     * @brief Prepare the multibody CCD system for collision detection
     * @tparam TDerivedX Eigen type of the input vertex positions
     * @tparam TDerivedVP Eigen type of the input vertex prefix sum
     * @tparam TDerivedEP Eigen type of the input edge prefix sum
     * @tparam TDerivedFP Eigen type of the input face prefix sum
     * @tparam TDerivedTP Eigen type of the input tetrahedron prefix sum
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param VP `|# objects + 1| x 1` prefix sum of vertex pointers into `V` s.t.
     * `V(VP(o):VP(o+1))` are collision vertices of object `o`
     * @param V `|# vertices|` vertex array
     * @param EP `|# objects + 1| x 1` prefix sum of edge pointers into `E` s.t. `E(EP(o):EP(o+1))`
     * are collision edges of object `o`
     * @param E `2 x |# edges|` edge array
     * @param FP `|# objects + 1| x 1` prefix sum of triangle pointers into `F` s.t.
     * `F(FP(o):FP(o+1))` are collision triangles of object `o`
     * @param F `3 x |# triangles|` triangle array
     * @param TP `|# objects + 1| x 1` prefix sum of tetrahedron pointers into `T` s.t.
     * `T(TP(o):TP(o+1))` are collision tetrahedra of object `o`
     * @param T `4 x |# tetrahedra|` tetrahedron array
     */
    template <
        class TDerivedX,
        class TDerivedVP,
        class TDerivedEP,
        class TDerivedFP,
        class TDerivedTP>
    void Prepare(
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::DenseBase<TDerivedVP> const& VP,
        Eigen::Ref<IndexVectorX const> const& V,
        Eigen::DenseBase<TDerivedEP> const& EP,
        Eigen::Ref<IndexMatrix<2, Eigen::Dynamic> const> const& E,
        Eigen::DenseBase<TDerivedFP> const& FP,
        Eigen::Ref<IndexMatrix<3, Eigen::Dynamic> const> const& F,
        Eigen::DenseBase<TDerivedTP> const& TP,
        Eigen::Ref<IndexMatrix<4, Eigen::Dynamic> const> const& T);
    /**
     * @brief Update the active set of vertex-triangle and edge-edge contact pairs
     *
     * Finds all nearest vertex-triangle pairs for active vertices and adds them to the
     * corresponding active set.
     *
     * Finds all earliest vertex-triangle and edge-edge intersections in the linear trajectory
     * \f$ \mathbf{x} = (1-\Delta t) \mathbf{x}(t) + \Delta t \mathbf{x}(t+1) \; \forall \; \Delta t
     * \in [0,1]\f$ for inactive vertices and adds them to the corresponding active sets.
     *
     * @tparam FOnVertexTriangleContactPair Callback function with signature
     * `void(VertexTriangleContact c)`
     * @tparam FOnEdgeEdgeContactPair Callback function with signature `void(EdgeEdgeContact c)`
     * @tparam TDerivedXT Eigen type of vertex positions at time t
     * @tparam TDerivedX Eigen type of vertex positions at time t+1
     * @param XT `kDims x |# vertices|` matrix of vertex positions at time t
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     * @param XK `kDims x |# vertices|` matrix of current vertex positions
     * @param fOnVertexTriangleContactPair Callback function for vertex-triangle contact pairs
     * @param fOnEdgeEdgeContactPair Callback function for edge-edge contact pairs
     */
    template <
        class FOnVertexTriangleContactPair,
        class FOnEdgeEdgeContactPair,
        class TDerivedXT,
        class TDerivedX,
        class TDerivedXK>
    void UpdateActiveSet(
        Eigen::DenseBase<TDerivedXT> const& XT,
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::DenseBase<TDerivedXK> const& XK,
        FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair,
        FOnEdgeEdgeContactPair&& fOnEdgeEdgeContactPair);
    /**
     * @brief Update the active set using DCD
     *
     * @tparam FOnVertexTriangleContactPair Callback function with signature
     * `void(VertexTriangleContact c)`
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param fOnVertexTriangleContactPair Callback function for vertex-triangle contact pairs
     */
    template <class FOnVertexTriangleContactPair, class TDerivedX>
    void UpdateDcdActiveSet(
        Eigen::DenseBase<TDerivedX> const& X,
        FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair);
    /**
     * @brief Update the active set using CCD
     *
     * @tparam FOnVertexTriangleContactPair Callback function with signature
     * `void(VertexTriangleContact c)`
     * @tparam FOnEdgeEdgeContactPair Callback function with signature `void(EdgeEdgeContact c)`
     * @tparam TDerivedXT Eigen type of vertex positions at time t
     * @tparam TDerivedX Eigen type of vertex positions at time t+1
     * @param XT `kDims x |# vertices|` matrix of vertex positions at time t
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     * @param fOnVertexTriangleContactPair Callback function for vertex-triangle contact pairs
     * @param fOnEdgeEdgeContactPair Callback function for edge-edge contact pairs
     */
    template <
        class FOnVertexTriangleContactPair,
        class FOnEdgeEdgeContactPair,
        class TDerivedXT,
        class TDerivedX>
    void UpdateCcdActiveSet(
        Eigen::DenseBase<TDerivedXT> const& XT,
        Eigen::DenseBase<TDerivedX> const& X,
        FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair,
        FOnEdgeEdgeContactPair&& fOnEdgeEdgeContactPair);

  protected:
    /**
     * @brief Add DCD vertex-triangle and edge-triangle pairs to the active set
     * @tparam FOnVertexTriangleContactPair Callback function with signature
     * `void(VertexTriangleContact c)`
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param fOnVertexTriangleContactPair Callback function for vertex-triangle contact pairs
     */
    template <class FOnVertexTriangleContactPair, class TDerivedX>
    void HandleDcdPairs(
        Eigen::DenseBase<TDerivedX> const& X,
        FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair);
    /**
     * @brief Add CCD vertex-triangle and edge-edge pairs to the active set
     *
     * @tparam FOnVertexTriangleContactPair Callback function with signature
     * `void(VertexTriangleContact c)`
     * @tparam FOnEdgeEdgeContactPair Callback function with signature `void(EdgeEdgeContact c)`
     * @tparam TDerivedXT Eigen type of vertex positions at time t
     * @tparam TDerivedX Eigen type of vertex positions at time t+1
     * @param XT `kDims x |# vertices|` matrix of vertex positions at time t
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     * @param fOnVertexTriangleContactPair Callback function for vertex-triangle contact pairs
     * @param fOnEdgeEdgeContactPair Callback function for edge-edge contact pairs
     */
    template <
        class FOnVertexTriangleContactPair,
        class FOnEdgeEdgeContactPair,
        class TDerivedXT,
        class TDerivedX>
    void HandleCcdPairs(
        Eigen::DenseBase<TDerivedXT> const& XT,
        Eigen::DenseBase<TDerivedX> const& X,
        FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair,
        FOnEdgeEdgeContactPair&& fOnEdgeEdgeContactPair);
    /**
     * @brief Compute axis-aligned bounding boxes for vertices
     *
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void ComputeVertexAabbs(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Compute axis-aligned bounding boxes for linearly swept vertices
     *
     * @tparam TDerivedX Eigen type of vertex positions
     * @param XT `kDims x |# vertices|` matrix of vertex positions at time t
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     */
    template <class TDerivedXT, class TDerivedX>
    void ComputeVertexAabbs(
        Eigen::DenseBase<TDerivedXT> const& XT,
        Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Compute axis-aligned bounding boxes for linearly swept edges
     * @tparam TDerivedX Eigen type of vertex positions
     * @param XT `kDims x |# vertices|` matrix of vertex positions at time t
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     */
    template <class TDerivedXT, class TDerivedX>
    void
    ComputeEdgeAabbs(Eigen::DenseBase<TDerivedXT> const& XT, Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Compute axis-aligned bounding boxes for triangles for DCD
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void ComputeTriangleAabbs(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Compute axis-aligned bounding boxes for linearly swept triangles
     * @tparam TDerivedX Eigen type of vertex positions
     * @param XT `kDims x |# vertices|` matrix of vertex positions at time t
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     */
    template <class TDerivedXT, class TDerivedX>
    void ComputeTriangleAabbs(
        Eigen::DenseBase<TDerivedXT> const& XT,
        Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Compute axis-aligned bounding boxes for tetrahedra
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void ComputeTetrahedronAabbs(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Computes body AABBs from mesh vertex BVHs
     * @pre (Vertex) mesh BVHs must be up-to-date before calling this function, i.e. via a call to
     * `UpdateMeshVertexBvhs`
     */
    void ComputeBodyAabbs();
    /**
     * @brief Recompute mesh vertex BVH bounding boxes
     */
    void UpdateMeshVertexBvhs();
    /**
     * @brief Recompute mesh edge BVH bounding boxes
     */
    void UpdateMeshEdgeBvhs();
    /**
     * @brief Recompute mesh triangle BVH bounding boxes
     */
    void UpdateMeshTriangleBvhs();
    /**
     * @brief Recompute mesh tetrahedron BVH bounding boxes
     */
    void UpdateMeshTetrahedronBvhs();
    /**
     * @brief Recompute body BVH tree and internal node bounding boxes
     */
    void RecomputeBodyBvh();
    /**
     * @brief Loop over all potentially colliding body pairs
     *
     * @tparam FOnBodyPair Callable with signature `void(Index oi, Index oj)`
     * @param fOnBodyPair Callback function for body pairs
     * @pre Body BVH must be up-to-date.
     */
    template <class FOnBodyPair>
    void ForEachBodyPair(FOnBodyPair&& fOnBodyPair);
    /**
     * @brief Loop over penetrating vertices
     *
     * @tparam FOnPenetratingVertex Callable with signature `void(Index v, Index o)`
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param fOnPenetratingVertex Callback function for penetrating vertices
     * @pre Vertex, tetrahedron and body BVHs must be up-to-date.
     */
    template <class FOnPenetratingVertex, class TDerivedX>
    void ForEachPenetratingVertex(
        Eigen::DenseBase<TDerivedX> const& X,
        FOnPenetratingVertex&& fOnPenetratingVertex);
    /**
     * @brief
     *
     * @tparam FOnVertexTriangleContactPair
     * @tparam TDerivedXT
     * @tparam TDerivedX
     * @param XT
     * @param X
     * @param oi
     * @param oj
     * @param fOnVertexTriangleContactPair
     */
    template <class FOnVertexTriangleContactPair, class TDerivedXT, class TDerivedX>
    void ReportVertexTriangleCcdContacts(
        Eigen::DenseBase<TDerivedXT> const& XT,
        Eigen::DenseBase<TDerivedX> const& X,
        Index oi,
        Index oj,
        FOnVertexTriangleContactPair fOnVertexTriangleContactPair);
    /**
     * @brief
     *
     * @tparam FOnEdgeEdgeContactPair
     * @tparam TDerivedXT
     * @tparam TDerivedX
     * @param XT
     * @param X
     * @param oi
     * @param oj
     * @param fOnEdgeEdgeContactPair
     */
    template <class FOnEdgeEdgeContactPair, class TDerivedXT, class TDerivedX>
    void ReportEdgeEdgeCcdContacts(
        Eigen::DenseBase<TDerivedXT> const& XT,
        Eigen::DenseBase<TDerivedX> const& X,
        Index oi,
        Index oj,
        FOnEdgeEdgeContactPair&& fOnEdgeEdgeContactPair);

  private:
    /**
     * @brief Prefix sums over mesh primitives
     */

    IndexVectorX mVP; ///< `|# objects + 1|` prefix sum of vertex pointers into `V`
    IndexVectorX mEP; ///< `|# objects + 1|` prefix sum of edge pointers into `E`
    IndexVectorX mFP; ///< `|# objects + 1|` prefix sum of triangle pointers into `F`
    IndexVectorX mTP; ///< `|# objects + 1|` prefix sum of tetrahedron pointers into `T`

    /**
     * @brief Mesh primitives
     */

    Eigen::Ref<IndexVectorX const>
        mV; ///< Flattened `|# objects|` list of `|# collision verts|` vertex arrays
    Eigen::Ref<IndexMatrix<2, Eigen::Dynamic> const>
        mE; ///< Flattened `|# objects|` list of `2x|# collision edges|` edge arrays
    Eigen::Ref<IndexMatrix<3, Eigen::Dynamic> const>
        mF; ///< Flattened `|# objects|` list of `3x|# collision triangles|` triangle arrays
    Eigen::Ref<IndexMatrix<4, Eigen::Dynamic> const>
        mT; ///< Flattened `|# objects|` list of `4x|# tetrahedra|` tetrahedron arrays

    /**
     * @brief Mesh primitive AABBs
     */

    Matrix<2 * kDims, Eigen::Dynamic> mVertexAabbs; ///< Flattened `|# objects|` list of `2*kDims x
                                                    ///< |# vertices|` axis-aligned bounding boxes
    Matrix<2 * kDims, Eigen::Dynamic> mEdgeAabbs; ///< Flattened `|# objects|` list of `2*kDims x |#
                                                  ///< edges|` axis-aligned bounding boxes
    Matrix<2 * kDims, Eigen::Dynamic>
        mTriangleAabbs; ///< Flattened `|# objects|` list of `2*kDims x |# triangles|` axis-aligned
                        ///< bounding boxes
    Matrix<2 * kDims, Eigen::Dynamic>
        mTetrahedronAabbs; ///< Flattened `|# objects|` list of `2*kDims x |# tetrahedra|`
                           ///< axis-aligned bounding boxes

    /**
     * @brief Mesh BVHs
     */

    std::vector<VertexBvh> mVertexBvhs;           ///< `|# objects|` list of mesh vertex BVHs
    std::vector<EdgeBvh> mEdgeBvhs;               ///< `|# objects|` list of mesh edge BVHs
    std::vector<TriangleBvh> mTriangleBvhs;       ///< `|# objects|` list of mesh triangle BVHs
    std::vector<TetrahedronBvh> mTetrahedronBvhs; ///< `|# objects|` list of mesh tetrahedron BVHs

    /**
     * @brief Body AABBs and BVH
     */

    Matrix<2 * kDims, Eigen::Dynamic>
        mBodyAabbs;   ///< `2*kDims x |# objects|` axis-aligned bounding boxes over each object
    BodyBvh mBodyBvh; ///< BVH over all objects in the world

    Eigen::Vector<bool, Eigen::Dynamic> mPenetratingVertexMask; ///< Active vertex mask
};

template <class TDerivedX, class TDerivedVP, class TDerivedEP, class TDerivedFP, class TDerivedTP>
inline MultibodyMeshMixedCcdDcd::MultibodyMeshMixedCcdDcd(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedVP> const& VP,
    Eigen::Ref<IndexVectorX const> const& V,
    Eigen::DenseBase<TDerivedEP> const& EP,
    Eigen::Ref<IndexMatrix<2, Eigen::Dynamic> const> const& E,
    Eigen::DenseBase<TDerivedFP> const& FP,
    Eigen::Ref<IndexMatrix<3, Eigen::Dynamic> const> const& F,
    Eigen::DenseBase<TDerivedTP> const& TP,
    Eigen::Ref<IndexMatrix<4, Eigen::Dynamic> const> const& T)
{
    Prepare(
        X.derived(),
        VP.derived(),
        V.derived(),
        EP.derived(),
        E.derived(),
        FP.derived(),
        F.derived());
}

template <class TDerivedX, class TDerivedVP, class TDerivedEP, class TDerivedFP, class TDerivedTP>
inline void MultibodyMeshMixedCcdDcd::Prepare(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedVP> const& VP,
    Eigen::Ref<IndexVectorX const> const& V,
    Eigen::DenseBase<TDerivedEP> const& EP,
    Eigen::Ref<IndexMatrix<2, Eigen::Dynamic> const> const& E,
    Eigen::DenseBase<TDerivedFP> const& FP,
    Eigen::Ref<IndexMatrix<3, Eigen::Dynamic> const> const& F,
    Eigen::DenseBase<TDerivedTP> const& TP,
    Eigen::Ref<IndexMatrix<4, Eigen::Dynamic> const> const& T)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.Prepare");
    // Store input triangle meshes
    mVP = VP;
    mEP = EP;
    mFP = FP;
    mTP = TP;
    mV  = V;
    mE  = E;
    mF  = F;
    mT  = T;
    // Allocate memory for AABBs and active sets
    mVertexAabbs.resize(2 * kDims, V.cols());
    mEdgeAabbs.resize(2 * kDims, E.cols());
    mTriangleAabbs.resize(2 * kDims, F.cols());
    mTetrahedronAabbs.resize(2 * kDims, T.cols());
    mBodyAabbs.resize(2 * kDims, VP.size() - 1);
    mPenetratingVertexMask.resize(X.cols());
    mPenetratingVertexMask.setConstant(false);
    // Allocate memory for mesh BVHs
    auto const nObjects = mVP.size() - 1;
    mVertexBvhs.resize(nObjects);
    mEdgeBvhs.resize(nObjects);
    mTriangleBvhs.resize(nObjects);
    mTetrahedronBvhs.resize(nObjects);
    // Compute static mesh primitive AABBs for tree construction
    ComputeVertexAabbs(X.derived());
    ComputeEdgeAabbs(X.derived(), X.derived());
    ComputeTriangleAabbs(X.derived());
    ComputeTetrahedronAabbs(X.derived());
    // Construct mesh BVHs
    for (auto o = 0; o < nObjects; ++o)
    {
        auto VB = mVertexAabbs(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        auto EB = mEdgeAabbs(Eigen::placeholders::all, Eigen::seq(mEP(o), mEP(o + 1) - 1));
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
        auto TB = mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(mTP(o), mTP(o + 1) - 1));
        // Construct object o's mesh BVH tree topology
        mVertexBvhs[o].Construct(VB.topRows<kDims>(), VB.bottomRows<kDims>());
        mEdgeBvhs[o].Construct(EB.topRows<kDims>(), EB.bottomRows<kDims>());
        mTriangleBvhs[o].Construct(FB.topRows<kDims>(), FB.bottomRows<kDims>());
        mTetrahedronBvhs[o].Construct(TB.topRows<kDims>(), TB.bottomRows<kDims>());
        // Compute object o's AABB
        auto XVo = X(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        mBodyAabbs.col(o).head<kDims>() = XVo.rowwise().minCoeff();
        mBodyAabbs.col(o).tail<kDims>() = XVo.rowwise().maxCoeff();
    }
    // Construct body BVH to allocate memory
    mBodyBvh.Construct(mBodyAabbs);
}

template <
    class FOnVertexTriangleContactPair,
    class FOnEdgeEdgeContactPair,
    class TDerivedXT,
    class TDerivedX,
    class TDerivedXK>
inline void MultibodyMeshMixedCcdDcd::UpdateActiveSet(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedXK> const& XK,
    FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair,
    FOnEdgeEdgeContactPair&& fOnEdgeEdgeContactPair)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.UpdateActiveSet");
    UpdateDcdActiveSet(
        XK,
        std::forward<FOnVertexTriangleContactPair>(fOnVertexTriangleContactPair));
    UpdateCcdActiveSet(
        XT,
        X,
        std::forward<FOnVertexTriangleContactPair>(fOnVertexTriangleContactPair),
        std::forward<FOnEdgeEdgeContactPair>(fOnEdgeEdgeContactPair));
}

template <class FOnVertexTriangleContactPair, class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::UpdateDcdActiveSet(
    Eigen::DenseBase<TDerivedX> const& X,
    FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair)
{
    // Update AABBs for discrete collision detection
    ComputeVertexAabbs(X);
    ComputeTriangleAabbs(X);
    ComputeTetrahedronAabbs(X);
    // Update BVH bounding boxes
    UpdateMeshVertexBvhs();
    UpdateMeshTriangleBvhs();
    UpdateMeshTetrahedronBvhs();
    // Rebuild world tree
    ComputeBodyAabbs();
    RecomputeBodyBvh();
    // Add DCD vertex-triangle pairs to active set
    mPenetratingVertexMask.setConstant(false);
    HandleDcdPairs(X, std::forward<FOnVertexTriangleContactPair>(fOnVertexTriangleContactPair));
}

template <
    class FOnVertexTriangleContactPair,
    class FOnEdgeEdgeContactPair,
    class TDerivedXT,
    class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::UpdateCcdActiveSet(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X,
    FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair,
    FOnEdgeEdgeContactPair&& fOnEdgeEdgeContactPair)
{
    // Compute AABBs for linearly swept vertices, edges and triangles
    ComputeVertexAabbs(XT, X);
    ComputeEdgeAabbs(XT, X);
    ComputeTriangleAabbs(XT, X);
    // Update BVHs
    UpdateMeshVertexBvhs();
    UpdateMeshEdgeBvhs();
    UpdateMeshTriangleBvhs();
    // Rebuild world tree
    ComputeBodyAabbs();
    RecomputeBodyBvh();
    // Report all vertex-triangle and edge-edge CCD contacts
    HandleCcdPairs(
        XT,
        X,
        std::forward<FOnVertexTriangleContactPair>(fOnVertexTriangleContactPair),
        std::forward<FOnEdgeEdgeContactPair>(fOnEdgeEdgeContactPair));
}

template <class FOnVertexTriangleContactPair, class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::HandleDcdPairs(
    Eigen::DenseBase<TDerivedX> const& X,
    FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair)
{
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.HandleDcdPairs");
    // Report vertex-triangle contacts for each DCD penetrating vertex
    ForEachPenetratingVertex([&](Index i, Index o) {
        mPenetratingVertexMask[i] = true;
        // Contact point is vertex position
        Vector<kDims> const XC = X.col(i);
        // Define point-aabb and point-triangle distance functions for vertex i
        auto const fDistanceToBox = [&]<class TL, class TU>(TL const& L, TU const& U) {
            using math::linalg::mini::FromEigen;
            return geometry::DistanceQueries::PointAxisAlignedBoundingBox(
                FromEigen(XC),
                FromEigen(L),
                FromEigen(U));
        };
        auto const fDistanceToTriangle = [&](Index f) {
            using math::linalg::mini::FromEigen;
            Matrix<kDims, 3> XF = X(Eigen::placeholders::all, mF.col(f));
            return geometry::DistanceQueries::PointTriangle(
                FromEigen(XC),
                FromEigen(XF.col(0)),
                FromEigen(XF.col(1)),
                FromEigen(XF.col(2)));
        };
        // Find nearest surface point (i.e. triangle)
        mTriangleBvhs[o].NearestNeighbour(
            fDistanceToBox,
            fDistanceToTriangle,
            [&, fReport = std::forward<FOnVertexTriangleContactPair>(fOnVertexTriangleContactPair)](
                Index f,
                [[maybe_unused]] Scalar d,
                [[maybe_unused]] Index k) {
                using math::linalg::mini::FromEigen;
                f = mFP(o) + f;
                fReport(VertexTriangleContact{i, f, FromEigen(mF.col(f))});
            });
    });
#include "pbat/warning/Pop.h"
}

template <
    class FOnVertexTriangleContactPair,
    class FOnEdgeEdgeContactPair,
    class TDerivedXT,
    class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::HandleCcdPairs(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X,
    FOnVertexTriangleContactPair&& fOnVertexTriangleContactPair,
    FOnEdgeEdgeContactPair&& fOnEdgeEdgeContactPair)
{
    ForEachBodyPair([&](Index oi, Index oj) {
        ReportVertexTriangleCcdContacts(
            XT,
            X,
            oi,
            oj,
            std::forward<FOnVertexTriangleContactPair>(fOnVertexTriangleContactPair));
        ReportVertexTriangleCcdContacts(
            XT,
            X,
            oj,
            oi,
            std::forward<FOnVertexTriangleContactPair>(fOnVertexTriangleContactPair));
        ReportEdgeEdgeCcdContacts(
            XT,
            X,
            oi,
            oj,
            std::forward<FOnEdgeEdgeContactPair>(fOnEdgeEdgeContactPair));
    });
}

template <class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ComputeVertexAabbs(Eigen::DenseBase<TDerivedX> const& X)
{
    auto XV                          = X(Eigen::placeholders::all, mV);
    mVertexAabbs.topRows<kDims>()    = XV;
    mVertexAabbs.bottomRows<kDims>() = XV;
}

template <class TDerivedXT, class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ComputeVertexAabbs(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.ComputeVertexAabbs");
    auto XTV = XT.template topRows<kDims>()(Eigen::placeholders::all, mV);
    auto XV  = X.template topRows<kDims>()(Eigen::placeholders::all, mV);
    for (auto v = 0; v < mV.cols(); ++v)
    {
        auto i = mV(v);
        auto L = mVertexAabbs.col(v).head<kDims>();
        auto U = mVertexAabbs.col(v).tail<kDims>();
        L      = XTV.col(i).cwiseMin(XV.col(i));
        U      = XTV.col(i).cwiseMin(XV.col(i));
    }
}

template <class TDerivedXT, class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ComputeEdgeAabbs(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.ComputeEdgeAabbs");
    for (auto e = 0; e < mE.cols(); ++e)
    {
        Matrix<kDims, 2 * 2> XE;
        XE.leftCols<2>()  = XT(Eigen::placeholders::all, mE.col(e)).block<kDims, 2>(0, 0);
        XE.rightCols<2>() = X(Eigen::placeholders::all, mE.col(e)).block<kDims, 2>(0, 0);
        auto L            = mEdgeAabbs.col(e).head<kDims>();
        auto U            = mEdgeAabbs.col(e).tail<kDims>();
        L                 = XE.rowwise().minCoeff();
        U                 = XE.rowwise().maxCoeff();
    }
}

template <class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ComputeTriangleAabbs(Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.ComputeTriangleAabbs");
#include "pbat/warning/Push.h"
#include "pbat/warning/SignConversion.h"
    for (auto f = 0; f < mF.cols(); ++f)
    {
        Matrix<kDims, 3> XF;
        XF     = X(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0);
        auto L = mTriangleAabbs.col(f).head<kDims>();
        auto U = mTriangleAabbs.col(f).tail<kDims>();
        L      = XF.rowwise().minCoeff();
        U      = XF.rowwise().maxCoeff();
    }
#include "pbat/warning/Pop.h"
}

template <class TDerivedXT, class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ComputeTriangleAabbs(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.ComputeTriangleAabbs");
    for (auto f = 0; f < mF.cols(); ++f)
    {
        Matrix<kDims, 2 * 3> XF;
        XF.leftCols<3>()  = XT(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0);
        XF.rightCols<3>() = X(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0);
        auto L            = mTriangleAabbs.col(f).head<kDims>();
        auto U            = mTriangleAabbs.col(f).tail<kDims>();
        L                 = XF.rowwise().minCoeff();
        U                 = XF.rowwise().maxCoeff();
    }
}

template <class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ComputeTetrahedronAabbs(Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MultibodyMeshMixedCcdDcd.ComputeTetrahedronAabbs");
    for (auto t = 0; t < mT.cols(); ++t)
    {
        Matrix<kDims, 4> XT;
        XT     = X(Eigen::placeholders::all, mT.col(t)).block<kDims, 4>(0, 0);
        auto L = mTetrahedronAabbs.col(t).head<kDims>();
        auto U = mTetrahedronAabbs.col(t).tail<kDims>();
        L      = XT.rowwise().minCoeff();
        U      = XT.rowwise().maxCoeff();
    }
}

template <class FOnBodyPair>
inline void MultibodyMeshMixedCcdDcd::ForEachBodyPair(FOnBodyPair&& fOnBodyPair)
{
    mBodyBvh.SelfOverlaps(
        [&](Index o1, Index o2) {
            auto L1 = mBodyAabbs.col(o1).head<kDims>();
            auto U1 = mBodyAabbs.col(o1).tail<kDims>();
            auto L2 = mBodyAabbs.col(o2).head<kDims>();
            auto U2 = mBodyAabbs.col(o2).tail<kDims>();
            return geometry::OverlapQueries::AxisAlignedBoundingBoxes(L1, U1, L2, U2);
        },
        [fOnBodyPair = std::forward<FOnBodyPair>(
             fOnBodyPair)](Index oi, Index oj, [[maybe_unused]] Index k) { fOnBodyPair(oi, oj); });
}

template <class FOnPenetratingVertex, class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ForEachPenetratingVertex(
    Eigen::DenseBase<TDerivedX> const& X,
    FOnPenetratingVertex&& fOnPenetratingVertex)
{
    auto const fVisitPenetratingVerticesOf =
        [&,
         fOnPenetratingVertex =
             std::forward<FOnPenetratingVertex>(fOnPenetratingVertex)](Index oi, Index oj) {
            Index i;
            Vector<kDims> xi;
            mVertexBvhs[oi].Overlaps(
                mTetrahedronBvhs[oj],
                [&](Index v, Index t) {
                    // Global vertex and tetrahedron indices
                    v = mVP(oi) + v;
                    t = mTP(oj) + t;
                    // Run overlap test
                    i                    = mV(v);
                    xi                   = X.col(i);
                    IndexVector<4> tinds = mT.col(t);
                    Matrix<kDims, 4> XT  = X(Eigen::placeholders::all, tinds);
                    using math::linalg::mini::FromEigen;
                    return geometry::OverlapQueries::PointTetrahedron3D(
                        FromEigen(xi),
                        FromEigen(XT.col(0)),
                        FromEigen(XT.col(1)),
                        FromEigen(XT.col(2)),
                        FromEigen(XT.col(3)));
                },
                [&]([[maybe_unused]] Index v, [[maybe_unused]] Index t, Index k) {
                    if (k == 0)
                        fOnPenetratingVertex(i, oj);
                });
        };
    ForEachBodyPair([&](Index oi, Index oj) {
        fVisitPenetratingVerticesOf(oi, oj);
        fVisitPenetratingVerticesOf(oj, oi);
    });
}

template <class FOnVertexTriangleContactPair, class TDerivedXT, class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ReportVertexTriangleCcdContacts(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X,
    Index oi,
    Index oj,
    FOnVertexTriangleContactPair fOnVertexTriangleContactPair)
{
    mVertexBvhs[oi].Overlaps(
        mTriangleBvhs[oj],
        [&,
         fOnVertexTriangleContactPair = std::forward<FOnVertexTriangleContactPair>(
             fOnVertexTriangleContactPair)](Index v, Index f) {
            // Global vertex and triangle indices
            v       = mVP(oi) + v;
            f       = mFP(oj) + f;
            Index i = mV(v);
            if (mPenetratingVertexMask[i])
                return false;
            // Run inexact CCD
            Vector<kDims> XTV        = XT.col(i);
            Vector<kDims> XV         = X.col(i);
            IndexVector<kDims> finds = mF.col(f);
            Matrix<kDims, 3> XTF     = XT(Eigen::placeholders::all, finds);
            Matrix<kDims, 3> XF      = X(Eigen::placeholders::all, finds);
            using namespace math::linalg::mini;
            SVector<Scalar, 4> tbeta = geometry::PointTriangleCcd(
                FromEigen(XTV),
                FromEigen(XTF.col(0)),
                FromEigen(XTF.col(1)),
                FromEigen(XTF.col(2)),
                FromEigen(XV),
                FromEigen(XF.col(0)),
                FromEigen(XF.col(1)),
                FromEigen(XF.col(2)));
            // Report if contact found
            auto const t           = tbeta[0];
            bool const bIntersects = t >= Scalar(0);
            if (bIntersects)
            {
                // Report contact to caller
                fOnVertexTriangleContactPair(VertexTriangleContact{i, f, FromEigen(finds)});
            }
            return bIntersects;
        },
        [](auto, auto, auto) {});
}

template <class FOnEdgeEdgeContactPair, class TDerivedXT, class TDerivedX>
inline void MultibodyMeshMixedCcdDcd::ReportEdgeEdgeCcdContacts(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X,
    Index oi,
    Index oj,
    FOnEdgeEdgeContactPair&& fOnEdgeEdgeContactPair)
{
    mEdgeBvhs[oi].Overlaps(
        mEdgeBvhs[oj],
        [&,
         fOnEdgeEdgeContactPair =
             std::forward<FOnEdgeEdgeContactPair>(fOnEdgeEdgeContactPair)](Index ei, Index ej) {
            // Global edge indices
            ei = mEP(oi) + ei;
            ej = mEP(oj) + ej;
            // Global edge vertex indices
            IndexVector<2> eindsi = mE.col(ei);
            IndexVector<2> eindsj = mE.col(ej);
            // Do not report contact if any vertex is already penetrating
            if (mPenetratingVertexMask[eindsi[0]] or mPenetratingVertexMask[eindsi[1]] or
                mPenetratingVertexMask[eindsj[0]] or mPenetratingVertexMask[eindsj[1]])
                return false;
            // Run inexact CCD
            Matrix<kDims, 2> XTei = XT(Eigen::placeholders::all, eindsi);
            Matrix<kDims, 2> Xei  = X(Eigen::placeholders::all, eindsi);
            Matrix<kDims, 2> XTej = XT(Eigen::placeholders::all, eindsj);
            Matrix<kDims, 2> Xej  = X(Eigen::placeholders::all, eindsj);
            using namespace math::linalg::mini;
            SVector<Scalar, 3> tbeta = geometry::EdgeEdgeCcd(
                FromEigen(XTei.col(0)),
                FromEigen(XTei.col(1)),
                FromEigen(XTej.col(0)),
                FromEigen(XTej.col(1)),
                FromEigen(Xei.col(0)),
                FromEigen(Xei.col(1)),
                FromEigen(Xej.col(0)),
                FromEigen(Xej.col(1)));
            // Report if contact found
            auto const t           = tbeta[0];
            bool const bIntersects = t >= Scalar(0);
            if (bIntersects)
            {
                auto uv = tbeta.Slice<2, 1>(1, 0);
                fOnEdgeEdgeContactPair(
                    EdgeEdgeContact{ei, ej, FromEigen(eindsi), FromEigen(eindsj), FromEigen(uv)});
            }
            return bIntersects;
        },
        [](auto, auto, auto) {});
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MULTIBODYMESHMIXEDCCDDCD_H
