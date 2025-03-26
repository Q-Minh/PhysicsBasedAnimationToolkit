/**
 * @file MultibodyTriangleMeshMixedCcdDcd.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief This file contains a multibody (triangle) mesh continuous collision detection system.
 * @date 2025-03-25
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_CONTACT_MULTIBODYTRIANGLEMESHMIXEDCCDDCD_H
#define PBAT_SIM_CONTACT_MULTIBODYTRIANGLEMESHMIXEDCCDDCD_H

#include "pbat/Aliases.h"
#include "pbat/geometry/AabbKdTreeHierarchy.h"
#include "pbat/geometry/AabbRadixTreeHierarchy.h"

#include <vector>

namespace pbat::sim::contact {

/**
 * @brief Multibody triangle mesh continuous collision detection system
 */
class MultibodyTriangleMeshMixedCcdDcd
{
  public:
    static auto constexpr kDims = 3; ///< Number of spatial dimensions
    using BodyBvh               = geometry::AabbRadixTreeHierarchy<kDims>; ///< BVH over bodies
    using VertexBvh             = geometry::AabbKdTreeHierarchy<kDims>; ///< BVH over mesh vertices
    using EdgeBvh               = geometry::AabbKdTreeHierarchy<kDims>; ///< BVH over mesh edges
    using TriangleBvh           = geometry::AabbKdTreeHierarchy<kDims>; ///< BVH over mesh triangles

    /**
     * @brief Default constructor
     */
    MultibodyTriangleMeshMixedCcdDcd() = default;
    /**
     * @brief Construct a new TriangleMeshMultibodyCcd object from input triangle meshes
     *
     * @tparam TDerivedX Eigen type of the input vertex positions
     * @tparam TDerivedVP Eigen type of the input vertex prefix sum
     * @tparam TDerivedV Eigen type of the input vertices
     * @tparam TDerivedEP Eigen type of the input edge prefix sum
     * @tparam TDerivedE Eigen type of the input edges
     * @tparam TDerivedFP Eigen type of the input face prefix sum
     * @tparam TDerivedF Eigen type of the input faces
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param VP `|# objects + 1| x 1` prefix sum of vertex pointers into `V` s.t.
     * `V(VP(o):VP(o+1))` are collision vertices of object `o`
     * @param V `|# vertices|` vertex array
     * @param EP `|# objects + 1| x 1` prefix sum of edge pointers into `E` s.t.
     * `E(EP(o):EP(o+1))` are collision edges of object `o`
     * @param E `2 x |# edges|` edge array
     * @param FP `|# objects + 1| x 1` prefix sum of triangle pointers into `F` s.t.
     * `F(FP(o):FP(o+1))` are collision triangles of object `o`
     * @param F `3 x |# triangles|` triangle array
     */
    template <
        class TDerivedX,
        class TDerivedVP,
        class TDerivedV,
        class TDerivedEP,
        class TDerivedE,
        class TDerivedFP,
        class TDerivedF>
    MultibodyTriangleMeshMixedCcdDcd(
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::DenseBase<TDerivedVP> const& VP,
        Eigen::DenseBase<TDerivedV> const& V,
        Eigen::DenseBase<TDerivedEP> const& EP,
        Eigen::DenseBase<TDerivedE> const& E,
        Eigen::DenseBase<TDerivedFP> const& FP,
        Eigen::DenseBase<TDerivedF> const& F);
    /**
     * @brief Prepare the multibody CCD system for collision detection
     * @tparam TDerivedX Eigen type of the input vertex positions
     * @tparam TDerivedVP Eigen type of the input vertex prefix sum
     * @tparam TDerivedV Eigen type of the input vertices
     * @tparam TDerivedEP Eigen type of the input edge prefix sum
     * @tparam TDerivedE Eigen type of the input edges
     * @tparam TDerivedFP Eigen type of the input face prefix sum
     * @tparam TDerivedF Eigen type of the input faces
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
     */
    template <
        class TDerivedX,
        class TDerivedVP,
        class TDerivedV,
        class TDerivedEP,
        class TDerivedE,
        class TDerivedFP,
        class TDerivedF>
    void Prepare(
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::DenseBase<TDerivedVP> const& VP,
        Eigen::DenseBase<TDerivedV> const& V,
        Eigen::DenseBase<TDerivedEP> const& EP,
        Eigen::DenseBase<TDerivedE> const& E,
        Eigen::DenseBase<TDerivedFP> const& FP,
        Eigen::DenseBase<TDerivedF> const& F);
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
     * @tparam TDerivedXT Eigen type of vertex positions at time t
     * @tparam TDerivedX Eigen type of vertex positions at time t+1
     * @param XT `kDims x |# vertices|` matrix of vertex positions at time t
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     * @param XK `kDims x |# vertices|` matrix of current vertex positions
     */
    template <class TDerivedXT, class TDerivedX, class TDerivedXK>
    void UpdateActiveSet(
        Eigen::DenseBase<TDerivedXT> const& XT,
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::DenseBase<TDerivedXK> const& XK);
    /**
     * @brief Finalize the active set of vertex-triangle and edge-edge contact pairs
     *
     * - Removes all vertex-triangle and edge-edge contact pairs that are resolved from the active
     * set.
     * - Marks all active vertices as inactive if their signed distance to the nearest triangle is
     * positive.
     * - Marks all vertices from unresolved active vertex-triangle contact pairs as active.
     *
     * @tparam TDerivedX Eigen type of vertex positions at time t+1
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     */
    template <class TDerivedX>
    void FinalizeActiveSet(Eigen::DenseBase<TDerivedX> const& X);

  protected:
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
     * @brief Compute axis-aligned bounding boxes for triangles
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param bForDcd Flag to indicate if the computation is for DCD
     */
    template <class TDerivedX>
    void ComputeTriangleAabbs(Eigen::DenseBase<TDerivedX> const& X, bool bForDcd = false);
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
     * @brief Computes body AABBs from mesh vertex BVHs
     * @pre (Vertex) mesh BVHs must be up-to-date before calling this function, i.e. via a call to
     * `UpdateMeshVertexBvhs`
     */
    void UpdateBodyAabbs();
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
     * @param bForDcd Flag to indicate if the computation is for DCD
     */
    void UpdateMeshTriangleBvhs(bool bForDcd = false);
    /**
     * @brief Recompute body BVH tree and internal node bounding boxes
     */
    void RecomputeBodyBvh();

  private:
    /**
     * @brief Prefix sums over mesh primitives
     */

    IndexVectorX mVP; ///< `|# objects + 1|` prefix sum of vertex pointers into `V`
    IndexVectorX mEP; ///< `|# objects + 1|` prefix sum of edge pointers into `E`
    IndexVectorX mFP; ///< `|# objects + 1|` prefix sum of triangle pointers into `F`

    /**
     * @brief Mesh primitives
     */

    IndexVectorX mV; ///< Flattened `|# objects|` list of `|# collision verts|` vertex arrays
    IndexMatrix<2, Eigen::Dynamic>
        mE; ///< Flattened `|# objects|` list of `2x|# collision edges|` edge arrays
    IndexMatrix<3, Eigen::Dynamic>
        mF; ///< Flattened `|# objects|` list of `3x|# collision triangles|` triangle arrays

    /**
     * @brief Mesh primitive AABBs
     */

    Matrix<2 * kDims, Eigen::Dynamic>
        mVertexAabbs; ///< Flattened `|# objects|` list of `2*kDims x |#
    ///< vertices|` axis-aligned bounding boxes
    Matrix<2 * kDims, Eigen::Dynamic> mEdgeAabbs; ///< Flattened `|# objects|` list of `2*kDims x |#
    ///< edges|` axis-aligned bounding boxes
    Matrix<2 * kDims, Eigen::Dynamic>
        mTriangleAabbs; ///< Flattened `|# objects|` list of `2*kDims x |#
    ///< triangles|` axis-aligned bounding boxes

    /**
     * @brief Mesh BVHs
     */

    std::vector<VertexBvh> mVertexBvhs;     ///< `|# objects|` list of mesh vertex BVHs
    std::vector<EdgeBvh> mEdgeBvhs;         ///< `|# objects|` list of mesh edge BVHs
    std::vector<TriangleBvh> mTriangleBvhs; ///< `|# objects|` list of mesh triangle BVHs

    /**
     * @brief Body AABBs and BVH
     */

    Matrix<2 * kDims, Eigen::Dynamic>
        mBodyAabbs;   ///< `2*kDims x |# objects|` axis-aligned bounding boxes over each object
    BodyBvh mBodyBvh; ///< BVH over all objects in the world

    /**
     * @brief Active set
     */
    Eigen::Vector<bool, Eigen::Dynamic>
        mIsVertexActive;                      ///< `|# collision vertices|` mask of active vertices
    IndexVectorX mActiveVertexTrianglePrefix; ///< `|# collision vertices|` prefix sum of active
                                              ///< vertex-triangle contact pairs `(v,f)`, s.t.
                                              ///< `mActiveTriangles[mActiveVertexTrianglePrefix(v),
                                              ///< mActiveVertexTrianglePrefix(v+1))` are active
                                              ///< triangles `f`
    IndexVectorX mActiveTriangles; ///< `|# active vertex-triangle pairs|` list of triangles `f`
                                   ///< from active vertex-triangle pairs `(v,f)`
    IndexVectorX mActiveEdgeEdgePrefix; ///< `|# collision edges|` prefix sum of active edge-edge
                                        ///< contact pairs `(ei,ej)`, s.t.
                                        ///< `mActiveEdges[mActiveEdgeEdgePrefix(ei),
                                        ///< mActiveEdgeEdgePrefix(ei+1))` are active edges `ej`
    IndexVectorX mActiveEdges; ///< `|# active edge-edge pairs|` list of edges `ej` from active
                               ///< edge-edge pairs `(ei,ej)`

    /**
     * @brief DCD
     *
     * We store stream-compacted penetrating vertices and penetrated bodies between time steps to
     * trim down DCD to only those vertices and bodies that are in contact.
     */

    Index mNDcdVertices; ///< Number of active vertices
    IndexMatrix<2, Eigen::Dynamic>
        mDcdVertexBodyPairs; ///< `2 x |# penetrating vertices|` list (v,o) of active vertices v
                             ///< penetrating object o
    Index mNDcdBodies;       ///< Number of penetrated bodies
    IndexVectorX mDcdBodies; ///< `|# penetrated bodies|` list of penetrated bodies
};

template <
    class TDerivedX,
    class TDerivedVP,
    class TDerivedV,
    class TDerivedEP,
    class TDerivedE,
    class TDerivedFP,
    class TDerivedF>
inline MultibodyTriangleMeshMixedCcdDcd::MultibodyTriangleMeshMixedCcdDcd(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedVP> const& VP,
    Eigen::DenseBase<TDerivedV> const& V,
    Eigen::DenseBase<TDerivedEP> const& EP,
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::DenseBase<TDerivedFP> const& FP,
    Eigen::DenseBase<TDerivedF> const& F)
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

template <
    class TDerivedX,
    class TDerivedVP,
    class TDerivedV,
    class TDerivedEP,
    class TDerivedE,
    class TDerivedFP,
    class TDerivedF>
inline void MultibodyTriangleMeshMixedCcdDcd::Prepare(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedVP> const& VP,
    Eigen::DenseBase<TDerivedV> const& V,
    Eigen::DenseBase<TDerivedEP> const& EP,
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::DenseBase<TDerivedFP> const& FP,
    Eigen::DenseBase<TDerivedF> const& F)
{
    // Store input triangle meshes
    mVP = VP;
    mEP = EP;
    mFP = FP;
    mV  = V;
    mE  = E;
    mF  = F;
    // Allocate memory for AABBs and active sets
    mVertexAabbs.resize(2 * kDims, V.cols());
    mEdgeAabbs.resize(2 * kDims, E.cols());
    mTriangleAabbs.resize(2 * kDims, F.cols());
    mBodyAabbs.resize(2 * kDims, VP.size() - 1);
    mIsVertexActive.resize(V.cols());
    mIsVertexActive.setConstant(false);
    mActiveVertexTrianglePrefix.resize(V.cols() + 1);
    mActiveVertexTrianglePrefix.setZero();
    mActiveTriangles.resize(F.cols() /*reasonable preallocation*/);
    mActiveEdgeEdgePrefix.resize(E.cols() + 1);
    mActiveEdgeEdgePrefix.setZero();
    mActiveEdges.resize(E.cols() /*reasonable preallocation*/);
    mNDcdVertices = 0;
    mDcdVertexBodyPairs.resize(2, V.cols() /*reasonable preallocation*/);
    mNDcdBodies = 0;
    mDcdBodies.resize(VP.size() - 1 /* # objects */);
    // Allocate memory for mesh BVHs
    auto const nObjects = mVP.size() - 1;
    mVertexBvhs.resize(nObjects);
    mEdgeBvhs.resize(nObjects);
    mTriangleBvhs.resize(nObjects);
    // Compute static mesh primitive AABBs for tree construction
    ComputeVertexAabbs(X.derived(), X.derived());
    ComputeEdgeAabbs(X.derived(), X.derived());
    ComputeTriangleAabbs(X.derived(), false /*bForDcd*/);
    // Construct mesh BVHs
    for (auto o = 0; o < nObjects; ++o)
    {
        auto VB = mVertexAabbs(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        auto EB = mEdgeAabbs(Eigen::placeholders::all, Eigen::seq(mEP(o), mEP(o + 1) - 1));
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
        // Construct object o's mesh BVH tree topology
        mVertexBvhs[o].Construct(VB.topRows<kDims>(), VB.bottomRows<kDims>());
        mEdgeBvhs[o].Construct(EB.topRows<kDims>(), EB.bottomRows<kDims>());
        mTriangleBvhs[o].Construct(FB.topRows<kDims>(), FB.bottomRows<kDims>());
        // Compute object o's AABB
        auto XVo = X(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        mBodyAabbs.col(o).head<kDims>() = XVo.rowwise().minCoeff();
        mBodyAabbs.col(o).tail<kDims>() = XVo.rowwise().maxCoeff();
    }
    // Construct body BVH to allocate memory
    mBodyBvh.Construct(mBodyAabbs);
}

template <class TDerivedXT, class TDerivedX, class TDerivedXK>
inline void MultibodyTriangleMeshMixedCcdDcd::UpdateActiveSet(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedXK> const& XK)
{
    // 1. Perform DCD for active vertex-triangle pairs (v,f,vo,fo), where v is body vo's vertex and
    // f is body fo's triangle.

    // a) Recompute triangle AABBs of penetrated bodies of fo
    ComputeTriangleAabbs(XK, true /*bForDcd*/);
    // b) Update BVH bounding boxes of meshes fo
    UpdateMeshTriangleBvhs(true /*bForDcd*/);
    // c) Compute nearest (active) vertex-triangle pairs (v,f) and add them to the active set

    // 2. Perform CCD for (inactive) vertex-triangle and edge-edge pairs.

    // a) Compute AABBs for linearly swept vertices, edges and triangles
    ComputeVertexAabbs(XT, X);
    ComputeEdgeAabbs(XT, X);
    ComputeTriangleAabbs(XT, X);
    // b) Update mesh BVH bounding boxes
    UpdateMeshVertexBvhs();
    UpdateMeshEdgeBvhs();
    UpdateMeshTriangleBvhs();
    // c) Recompute body BVH tree and internal node bounding boxes
    UpdateBodyAabbs();
    RecomputeBodyBvh();
    // d) Find potentially contacting body pairs (bi,bj)

    // e) For each body pair (bi,bj), find all earliest (inactive) vertex-triangle and edge-edge
    // intersections and add them to the active set
}

template <class TDerivedXT, class TDerivedX>
inline void MultibodyTriangleMeshMixedCcdDcd::ComputeVertexAabbs(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X)
{
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
inline void MultibodyTriangleMeshMixedCcdDcd::ComputeEdgeAabbs(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X)
{
    for (auto e = 0; e < mE.cols(); ++e)
    {
        Matrix<kDims, 2 * 2> XE;
        XE.leftCols<2>()  = XT(Eigen::placeholders::all, mE.col(e)).block<kDims, 2>(0, 0).eval();
        XE.rightCols<2>() = X(Eigen::placeholders::all, mE.col(e)).block<kDims, 2>(0, 0).eval();
        auto L            = mEdgeAabbs.col(e).head<kDims>();
        auto U            = mEdgeAabbs.col(e).tail<kDims>();
        L                 = XE.rowwise().minCoeff();
        U                 = XE.rowwise().maxCoeff();
    }
}

template <class TDerivedX>
inline void MultibodyTriangleMeshMixedCcdDcd::ComputeTriangleAabbs(
    Eigen::DenseBase<TDerivedX> const& X,
    bool bForDcd)
{
    if (bForDcd)
    {
        for (auto k = 0; k < mNDcdBodies; ++k)
        {
            auto o     = mDcdBodies(k);
            auto begin = mFP(o);
            auto end   = mFP(o + 1);
            for (auto f = begin; f < end; ++f)
            {
                auto XF = X(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0).eval();
                auto L  = mTriangleAabbs.col(f).head<kDims>();
                auto U  = mTriangleAabbs.col(f).tail<kDims>();
                L       = XF.rowwise().minCoeff();
                U       = XF.rowwise().maxCoeff();
            }
        }
    }
    else
    {
        for (auto f = 0; f < mF.cols(); ++f)
        {
            auto XF = X(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0).eval();
            auto L  = mTriangleAabbs.col(f).head<kDims>();
            auto U  = mTriangleAabbs.col(f).tail<kDims>();
            L       = XF.rowwise().minCoeff();
            U       = XF.rowwise().maxCoeff();
        }
    }
}

template <class TDerivedXT, class TDerivedX>
inline void MultibodyTriangleMeshMixedCcdDcd::ComputeTriangleAabbs(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X)
{
    for (auto f = 0; f < mF.cols(); ++f)
    {
        Matrix<kDims, 2 * 3> XF;
        XF.leftCols<3>()  = XT(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0).eval();
        XF.rightCols<3>() = X(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0).eval();
        auto L            = mTriangleAabbs.col(f).head<kDims>();
        auto U            = mTriangleAabbs.col(f).tail<kDims>();
        L                 = XF.rowwise().minCoeff();
        U                 = XF.rowwise().maxCoeff();
    }
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MULTIBODYTRIANGLEMESHMIXEDCCDDCD_H
