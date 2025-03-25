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
     * @param VP `|# objects + 1|` prefix sum of vertex pointers into `V` s.t. `V(VP(o):VP(o+1))`
     * are collision vertices of object `o`
     * @param V `|# vertices|` vertex array
     * @param EP `|# objects + 1|` prefix sum of edge pointers into `E` s.t. `E(EP(o):EP(o+1))`
     * are collision edges of object `o`
     * @param E `|# edges|` edge array
     * @param FP `|# objects + 1|` prefix sum of triangle pointers into `F` s.t. `F(FP(o):FP(o+1))`
     * are collision triangles of object `o`
     * @param F `|# triangles|` triangle array
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
     * @param VP `|# objects + 1|` prefix sum of vertex pointers into `V` s.t. `V(VP(o):VP(o+1))`
     * are collision vertices of object `o`
     * @param V `|# vertices|` vertex array
     * @param EP `|# objects + 1|` prefix sum of edge pointers into `E` s.t. `E(EP(o):EP(o+1))` are
     * collision edges of object `o`
     * @param E `|# edges|` edge array
     * @param FP `|# objects + 1|` prefix sum of triangle pointers into `F` s.t. `F(FP(o):FP(o+1))`
     * are collision triangles of object `o`
     * @param F `|# triangles|` triangle array
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

  protected:
    /**
     * @brief Compute axis-aligned bounding boxes for linearly swept vertices, edges and triangles
     *
     * @tparam TDerivedXT Eigen type of vertex positions at time t
     * @tparam TDerivedX Eigen type of vertex positions at time t+1
     * @param XT `kDims x |# vertices|` matrix of vertex positions at time t
     * @param X `kDims x |# vertices|` matrix of vertex positions at time t+1
     */
    template <class TDerivedXT, class TDerivedX>
    void ComputeAabbs(Eigen::DenseBase<TDerivedXT> const& XT, Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Compute axis-aligned bounding boxes for vertices, edges and triangles
     *
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void ComputeAabbs(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Recompute mesh BVH bounding boxes
     */
    void UpdateMeshBvhs();
    /**
     * @brief Recompute body BVH tree and internal node bounding boxes
     */
    void RecomputeBodyBvh();

  private:
    IndexVectorX mVP; ///< `|# objects + 1|` prefix sum of vertex pointers into `V`
    IndexVectorX mEP; ///< `|# objects + 1|` prefix sum of edge pointers into `E`
    IndexVectorX mFP; ///< `|# objects + 1|` prefix sum of triangle pointers into `F`

    IndexVectorX mV; ///< Flattened `|# objects|` list of `|# collision verts|` vertex arrays
    IndexMatrix<2, Eigen::Dynamic>
        mE; ///< Flattened `|# objects|` list of `2x|# collision edges|` edge arrays
    IndexMatrix<3, Eigen::Dynamic>
        mF; ///< Flattened `|# objects|` list of `3x|# collision triangles|` triangle arrays

    Matrix<kDims, Eigen::Dynamic> mVertexAabbs; ///< Flattened `|# objects|` list of `2*kDims x |#
    ///< vertices|` axis-aligned bounding boxes
    Matrix<kDims, Eigen::Dynamic> mEdgeAabbs; ///< Flattened `|# objects|` list of `2*kDims x |#
    ///< edges|` axis-aligned bounding boxes
    Matrix<kDims, Eigen::Dynamic> mTriangleAabbs; ///< Flattened `|# objects|` list of `2*kDims x |#
    ///< triangles|` axis-aligned bounding boxes

    std::vector<VertexBvh> mVertexBvhs;     ///< `|# objects|` list of mesh vertex BVHs
    std::vector<EdgeBvh> mEdgeBvhs;         ///< `|# objects|` list of mesh edge BVHs
    std::vector<TriangleBvh> mTriangleBvhs; ///< `|# objects|` list of mesh triangle BVHs

    Matrix<kDims, Eigen::Dynamic>
        mBodyAabbs;   ///< `2*kDims x |# objects|` axis-aligned bounding boxes over each object
    BodyBvh mBodyBvh; ///< BVH over all objects in the world
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
    : mVP(VP),
      mEP(EP),
      mFP(FP),
      mV(V),
      mE(E),
      mF(F),
      mVertexAabbs(kDims, V.cols()),
      mEdgeAabbs(kDims, E.cols()),
      mTriangleAabbs(kDims, F.cols()),
      mVertexBvhs(),
      mEdgeBvhs(),
      mTriangleBvhs(),
      mBodyAabbs(kDims, VP.size() - 1),
      mBodyBvh()
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
    // Allocate memory for spatial acceleration data structures
    auto const nObjects = mVP.size() - 1;
    mBodyAabbs.resize(kDims, nObjects);
    mVertexAabbs.resize(kDims, V.cols());
    mEdgeAabbs.resize(kDims, E.cols());
    mTriangleAabbs.resize(kDims, F.cols());
    // Compute static mesh primitive AABBs for tree construction
    ComputeAabbs(X.derived());
    // Construct mesh BVHs
    mVertexBvhs.resize(nObjects);
    mEdgeBvhs.resize(nObjects);
    mTriangleBvhs.resize(nObjects);
    for (auto o = 0; o < nObjects; ++o)
    {
        auto VB = mVertexAabbs(Eigen::placeholders::all, Eigen::seq(mVP(o), mVP(o + 1) - 1));
        auto EB = mEdgeAabbs(Eigen::placeholders::all, Eigen::seq(mEP(o), mEP(o + 1) - 1));
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
        // Construct object o's mesh BVH tree topology
        mVertexBvhs[o].Construct(VB);
        mEdgeBvhs[o].Construct(EB);
        mTriangleBvhs[o].Construct(FB);
        // Compute object o's AABB
        mBodyAabbs.col(o).head<kDims>() = VB.rowwise().minCoeff();
        mBodyAabbs.col(o).tail<kDims>() = VB.rowwise().maxCoeff();
    }
    // Construct body BVH
    mBodyBvh.Construct(mBodyAabbs);
}

template <class TDerivedXT, class TDerivedX>
inline void MultibodyTriangleMeshMixedCcdDcd::ComputeAabbs(
    Eigen::DenseBase<TDerivedXT> const& XT,
    Eigen::DenseBase<TDerivedX> const& X)
{
    for (auto v = 0; v < mV.cols(); ++v)
    {
        auto i   = mV(v);
        auto XTV = XT.col(i).head<kDims>();
        auto XV  = X.col(i).head<kDims>();
        auto L   = mVertexAabbs.col(v).head<kDims>();
        auto U   = mVertexAabbs.col(v).tail<kDims>();
        L        = XTV.cwiseMin(XV);
        U        = XTV.cwiseMax(XV);
    }
    for (auto e = 0; e < mE.cols(); ++e)
    {
        Matrix<kDims, 2 * 2> XE;
        XE.block<kDims, 2>(0, 0) = XT(Eigen::placeholders::all, mE.col(e));
        XE.block<kDims, 2>(0, 2) = X(Eigen::placeholders::all, mE.col(e));
        auto L                   = mEdgeAabbs.col(e).head<kDims>();
        auto U                   = mEdgeAabbs.col(e).tail<kDims>();
        L                        = XE.rowwise().minCoeff();
        U                        = XE.rowwise().maxCoeff();
    }
    for (auto f = 0; f < mF.cols(); ++f)
    {
        Matrix<kDims, 2 * 3> XF;
        XF.block<kDims, 3>(0, 0) = XT(Eigen::placeholders::all, mF.col(f));
        XF.block<kDims, 3>(0, 3) = X(Eigen::placeholders::all, mF.col(f));
        auto L                   = mTriangleAabbs.col(f).head<kDims>();
        auto U                   = mTriangleAabbs.col(f).tail<kDims>();
        L                        = XF.rowwise().minCoeff();
        U                        = XF.rowwise().maxCoeff();
    }
}

template <class TDerivedX>
inline void MultibodyTriangleMeshMixedCcdDcd::ComputeAabbs(Eigen::DenseBase<TDerivedX> const& X)
{
    auto XV                          = X(Eigen::placeholders::all, mV);
    mVertexAabbs.topRows<kDims>()    = XV;
    mVertexAabbs.bottomRows<kDims>() = XV;
    for (auto e = 0; e < mE.cols(); ++e)
    {
        auto XE = X(Eigen::placeholders::all, mE.col(e)).block<kDims, 2>(0, 0).eval();
        auto L  = mEdgeAabbs.col(e).head<kDims>();
        auto U  = mEdgeAabbs.col(e).tail<kDims>();
        L       = XE.rowwise().minCoeff();
        U       = XE.rowwise().maxCoeff();
    }
    for (auto f = 0; f < mF.cols(); ++f)
    {
        auto XF = X(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0).eval();
        auto L  = mTriangleAabbs.col(f).head<kDims>();
        auto U  = mTriangleAabbs.col(f).tail<kDims>();
        L       = XF.rowwise().minCoeff();
        U       = XF.rowwise().maxCoeff();
    }
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MULTIBODYTRIANGLEMESHMIXEDCCDDCD_H
