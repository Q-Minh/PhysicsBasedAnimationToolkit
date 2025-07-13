#ifndef PBAT_SIM_CONTACT_MESHVERTEXTETRAHEDRONDCD_H
#define PBAT_SIM_CONTACT_MESHVERTEXTETRAHEDRONDCD_H

#include "pbat/Aliases.h"
#include "pbat/geometry/AabbKdTreeHierarchy.h"
#include "pbat/geometry/AabbRadixTreeHierarchy.h"
#include "pbat/geometry/DistanceQueries.h"
#include "pbat/geometry/HierarchicalHashGrid.h"
#include "pbat/geometry/OverlapQueries.h"
#include "pbat/math/linalg/mini/Eigen.h"

#include <exception>
#include <unordered_map>
#include <vector>

namespace pbat::sim::contact {

class MeshVertexTetrahedronDcd
{
  public:
    using ScalarType = Scalar; ///< Scalar type used in the contact detection system.
    using IndexType  = Index;  ///< Index type used in the contact detection system.
    static constexpr int kMaxVertexTriangleContacts =
        8;                          ///< Maximum number of contacting triangles per vertex.
    static constexpr int kDims = 3; ///< Number of dimensions in the contact detection system.

    /**
     * @brief Construct a new MeshVertexTetrahedronDcd object from input tetrahedron meshes
     * @tparam TDerivedX Eigen type of the input vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param V `|# collision vertices|` vertex array
     * @param F `3 x |# triangles|` triangle array
     * @param T `4 x |# tetrahedra|` tetrahedron array
     * @param VP `|# objects + 1| x 1` prefix sum of vertex pointers into `V` s.t.
     * `V(VP(o):VP(o+1))` are collision vertices of object `o`
     * @param FP `|# objects + 1| x 1` prefix sum of triangle pointers into `F` s.t.
     * `F(FP(o):FP(o+1))` are collision triangles of object `o`
     * @param TP `|# objects + 1| x 1` prefix sum of tetrahedron pointers into `T` s.t.
     * `T(TP(o):TP(o+1))` are collision tetrahedra of object `o`
     */
    template <class TDerivedX>
    MeshVertexTetrahedronDcd(
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
        Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
        Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
        Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& TP);
    /**
     * @brief Update the active set of vertex-triangle contacts
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    [[maybe_unused]] void UpdateActiveSet(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Visit the triangles in active vertex-triangle contacts
     * @tparam FOnVertexTriangleContactPair Callable type with signature `void(
     * IndexType f, IndexType k)` where `f` is the index of the triangle and `0 <= k <
     * kMaxVertexTriangleContacts` is the index of the contact
     * @param v Index of the vertex whose active triangle contacts to visit
     * @param fOnVertexTriangleContactPair Function to call for each vertex-triangle contact pair
     */
    template <class FOnVertexTriangleContactPair>
    [[maybe_unused]] void ForEachActiveVertexTriangleContact(
        IndexType v,
        FOnVertexTriangleContactPair fOnVertexTriangleContactPair) const;
    /**
     * @brief Get the number of collison vertices in the system
     * @return Number of collision vertices
     */
    [[maybe_unused]] Eigen::Index GetNumVertices() const { return mV.size(); }
    /**
     * @brief Get the number of collision triangles in the system
     * @return Number of collision triangles
     */
    [[maybe_unused]] Eigen::Index GetNumTriangles() const { return mF.size(); }

  protected:
    /**
     * @brief Compute axis-aligned bounding boxes for triangles for DCD
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void ComputeTriangleAabbs(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Compute axis-aligned bounding boxes for tetrahedra
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void ComputeTetrahedronAabbs(Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Computes AABBs for tetrahedra from mesh vertex positions
     * @tparam TDerivedX Eigen type of vertex positions
     * @param o Index of the body to compute tetrahedron AABBs for
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void ComputeTetrahedronMeshAabbs(IndexType o, Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Computes body AABBs from mesh vertex BVHs
     * @pre (Vertex) mesh BVHs must be up-to-date before calling this function, i.e. via a call to
     * `UpdateMeshVertexBvhs`
     */
    void ComputeBodyAabbs();
    /**
     * @brief Recompute mesh triangle BVH bounding boxes
     */
    void UpdateMeshTriangleBvhs();
    /**
     * @brief Recompute mesh tetrahedron hash grid for a specific body
     * @param bodyIndex Index of the body to update the tetrahedron hash grid for
     */
    void UpdateMeshTetrahedronHashGrid(IndexType bodyIndex);
    /**
     * @brief Recompute bodies BVH tree and internal node bounding boxes
     */
    void RecomputeBodiesBvh();
    /**
     * @brief Compute and visit all contacting body pairs
     * @tparam FOnBodyPair Callable type with signature `void(IndexType oi, IndexType oj)`
     * @param fOnBodyPair Function to call for each body pair
     */
    template <class FOnBodyPair>
    void ForEachBodyPair(FOnBodyPair&& fOnBodyPair) const;
    /**
     * @brief Register all collision vertices of body `ov` that penetrate into body `ot`
     * @tparam TDerivedX Eigen type of vertex positions
     * @param ov Index of the body whose collision vertices to check against body `ot`
     * @param ot Index of the body penetrated by body `ov`
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void MarkPenetratingVertices(IndexType ov, IndexType ot, Eigen::DenseBase<TDerivedX> const& X);
    /**
     * @brief Find the nearest triangles on the triangle mesh `of` to penetrating vertices of body
     * `ov`
     * @tparam TDerivedX Eigen type of vertex positions
     * @param ov Index of the body whose collision vertices penetrate body `of`
     * @param of Index of the body penetrated by body `ov`
     * @param X `kDims x |# vertices|` matrix of vertex positions
     */
    template <class TDerivedX>
    void FindNearestTrianglesToPenetratingVertices(
        IndexType ov,
        IndexType of,
        Eigen::DenseBase<TDerivedX> const& X);

  private:
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const>
        mVP; ///< `|# objects + 1|` prefix sum of vertex pointers into `V`
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const>
        mFP; ///< `|# objects + 1|` prefix sum of triangle pointers into `F`
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const>
        mTP; ///< `|# objects + 1|` prefix sum of tetrahedron pointers into `T`

    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const>
        mV; ///< Flattened `|# objects|` list of `|# collision verts|` vertex arrays
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const>
        mF; ///< Flattened `|# objects|` list of `3x|# collision triangles|` triangle arrays
    Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const>
        mT; ///< Flattened `|# objects|` list of `4x|# tetrahedra|` tetrahedron arrays

    Eigen::Matrix<ScalarType, 2 * kDims, Eigen::Dynamic>
        mTriangleAabbs; ///< `|2*kDims| x |# triangles|` matrix s.t. mTriangleAabbs[:3,:]
                        ///< contains the AABB lowers and mTriangleAabbs[3:,:] contains the
                        ///< AABB uppers of each triangle.
    Eigen::Matrix<ScalarType, 2 * kDims, Eigen::Dynamic>
        mTetrahedronAabbs; ///< `|2*kDims| x |# tetrahedra|` matrix s.t. mTetrahedronAabbs[:3,:]
                           ///< contains the AABB lowers and mTetrahedronAabbs[3:,:] contains the
                           ///< AABB uppers of each tetrahedron.

    Eigen::Matrix<ScalarType, 2 * kDims, Eigen::Dynamic>
        mBodyAabbs; ///< `|2*kDims| x |# bodies|` matrix s.t. mBodyAabbs[:kDims,:] contains the AABB
                    ///< lowers and mBodyAabbs[kDims:,:] contains the AABB uppers of each body.

    std::vector<geometry::AabbKdTreeHierarchy<kDims>>
        mTriangleMeshBvhs; ///< Vector of k-D tree hierarchies over body triangle meshes.
    std::vector<geometry::HierarchicalHashGrid<kDims, ScalarType, IndexType>>
        mTetrahedronMeshHashGrids; ///< Hierarchical hash grid over tetrahedra.

    geometry::AabbRadixTreeHierarchy<kDims> mBodiesBvh; ///< Radix tree hierarchy over bodies.

    Eigen::Vector<bool, Eigen::Dynamic>
        mIsTetrahedronMeshHashGridDirty; ///< `|# bodies|` vector s.t.
                                         ///< mIsTetrahedronMeshHashGridDirty(i) is true if the
                                         ///< tetrahedron mesh hash grid for body i is dirty and
                                         ///< needs to be recomputed.
    std::vector<std::unordered_map<IndexType, IndexType>>
        mBodyVertexToOtherBodiesValence; ///< `|# bodies|` vector of hash maps s.t.
                                         ///< mBodyVertexToOtherBodiesValence[oi][vo] counts the
                                         ///< number of other bodies oj that vertex vo of body oi is
                                         ///< in contact with (that haven't been processed yet).
    Eigen::Matrix<IndexType, kMaxVertexTriangleContacts, Eigen::Dynamic>
        mVFC; ///< `|max # contacts| x |# vertices|` matrix s.t. mVFC(k, v) is the index of the k-th
              ///< contacting triangle with vertex v. If mVFC(k, v) == -1, there is no contact and
              ///< mVFC(k+j,v) == -1 for all j > 0.
};

template <class TDerivedX>
inline MeshVertexTetrahedronDcd::MeshVertexTetrahedronDcd(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& V,
    Eigen::Ref<Eigen::Matrix<IndexType, 3, Eigen::Dynamic> const> const& F,
    Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& VP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& FP,
    Eigen::Ref<Eigen::Vector<IndexType, Eigen::Dynamic> const> const& TP)
    : mVP(VP),
      mFP(FP),
      mTP(TP),
      mV(V),
      mF(F),
      mT(T),
      mTriangleAabbs(2 * kDims, F.cols()),
      mTetrahedronAabbs(2 * kDims, T.cols()),
      mBodyAabbs(2 * kDims, VP.size() - 1),
      mTriangleMeshBvhs(static_cast<std::size_t>(VP.size() - 1)),
      mTetrahedronMeshHashGrids(static_cast<std::size_t>(VP.size() - 1)),
      mBodiesBvh(),
      mIsTetrahedronMeshHashGridDirty(VP.size() - 1),
      mBodyVertexToOtherBodiesValence(static_cast<std::size_t>(VP.size() - 1)),
      mVFC(kMaxVertexTriangleContacts, V.size())
{
    // Compute initial AABBs
    ComputeTriangleAabbs(X.derived());
    ComputeTetrahedronAabbs(X.derived());
    // Construct spatial acceleration structures
    auto const nObjects = VP.size() - 1;
    for (auto o = 0; o < nObjects; ++o)
    {
        auto const oStl = static_cast<std::size_t>(o);
        auto FB = mTriangleAabbs(Eigen::placeholders::all, Eigen::seq(mFP(o), mFP(o + 1) - 1));
        auto TB = mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(mTP(o), mTP(o + 1) - 1));
        // Construct object o's mesh BVH tree topology
        mTriangleMeshBvhs[oStl].Construct(FB.topRows<kDims>(), FB.bottomRows<kDims>());
        auto const nTets = mTP(o + 1) - mTP(o);
        mTetrahedronMeshHashGrids[oStl].Configure(nTets);
    }
    ComputeBodyAabbs();
    mBodiesBvh.Construct(mBodyAabbs.topRows<kDims>(), mBodyAabbs.bottomRows<kDims>());
    mIsTetrahedronMeshHashGridDirty.setConstant(true);
    for (auto o = 0; o < nObjects; ++o)
    {
        auto const oStl = static_cast<std::size_t>(o);
        std::unordered_map<IndexType, IndexType>& bodyVertexToOtherBodiesValenceMap =
            mBodyVertexToOtherBodiesValence[oStl];
        auto vBegin = mVP(o);
        auto vEnd   = mVP(o + 1);
        bodyVertexToOtherBodiesValenceMap.reserve(static_cast<std::size_t>(vEnd - vBegin));
    }
    mVFC.setConstant(-1);
}

template <class TDerivedX>
inline void MeshVertexTetrahedronDcd::UpdateActiveSet(Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.UpdateActiveSet");
    mVFC.setConstant(-1);
    ComputeTriangleAabbs(X.derived());
    UpdateMeshTriangleBvhs();
    ComputeBodyAabbs();
    RecomputeBodiesBvh();
    ForEachBodyPair([&](IndexType oi, IndexType oj) {
        if (mIsTetrahedronMeshHashGridDirty(oi))
        {
            ComputeTetrahedronMeshAabbs(oi, X.derived());
            UpdateMeshTetrahedronHashGrid(oi);
            mIsTetrahedronMeshHashGridDirty(oi) = false;
        }
        if (mIsTetrahedronMeshHashGridDirty(oj))
        {
            ComputeTetrahedronMeshAabbs(oj, X.derived());
            UpdateMeshTetrahedronHashGrid(oj);
            mIsTetrahedronMeshHashGridDirty(oj) = false;
        }
        MarkPenetratingVertices(oi, oj, X.derived());
        MarkPenetratingVertices(oj, oi, X.derived());
        FindNearestTrianglesToPenetratingVertices(oi, oj, X.derived());
        FindNearestTrianglesToPenetratingVertices(oj, oi, X.derived());
    });
    mIsTetrahedronMeshHashGridDirty.setConstant(true);
}

template <class FOnVertexTriangleContactPair>
inline void MeshVertexTetrahedronDcd::ForEachActiveVertexTriangleContact(
    IndexType v,
    FOnVertexTriangleContactPair fOnVertexTriangleContactPair) const
{
    for (auto k = 0; k < kMaxVertexTriangleContacts; ++k)
    {
        IndexType const f = mVFC(k, v);
        if (f == -1)
            break;
        fOnVertexTriangleContactPair(f, k);
    }
}

template <class TDerivedX>
inline void MeshVertexTetrahedronDcd::ComputeTriangleAabbs(Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.ComputeTriangleAabbs");
    for (auto f = 0; f < mF.cols(); ++f)
    {
        Matrix<kDims, 3> XF;
        XF     = X(Eigen::placeholders::all, mF.col(f)).block<kDims, 3>(0, 0);
        auto L = mTriangleAabbs.col(f).head<kDims>();
        auto U = mTriangleAabbs.col(f).tail<kDims>();
        L      = XF.rowwise().minCoeff();
        U      = XF.rowwise().maxCoeff();
    }
}

template <class TDerivedX>
inline void MeshVertexTetrahedronDcd::ComputeTetrahedronAabbs(Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.ComputeTetrahedronAabbs");
    for (auto t = 0; t < mT.cols(); ++t)
    {
        Matrix<kDims, 4> XT = X(Eigen::placeholders::all, mT.col(t)).block<kDims, 4>(0, 0);
        auto L              = mTetrahedronAabbs.col(t).head<kDims>();
        auto U              = mTetrahedronAabbs.col(t).tail<kDims>();
        L                   = XT.rowwise().minCoeff();
        U                   = XT.rowwise().maxCoeff();
    }
}

template <class TDerivedX>
inline void MeshVertexTetrahedronDcd::ComputeTetrahedronMeshAabbs(
    IndexType o,
    Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.ComputeTetrahedronAabbs");
    auto const begin = mTP(o);
    auto const end   = mTP(o + 1);
    auto TB          = mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(begin, end - 1));
    for (auto t = begin; t < end; ++t)
    {
        Matrix<kDims, 4> XT = X(Eigen::placeholders::all, mT.col(t)).block<kDims, 4>(0, 0);
        auto L              = TB.col(t - begin).head<kDims>();
        auto U              = TB.col(t - begin).tail<kDims>();
        L                   = XT.rowwise().minCoeff();
        U                   = XT.rowwise().maxCoeff();
    }
}

template <class FOnBodyPair>
inline void MeshVertexTetrahedronDcd::ForEachBodyPair(FOnBodyPair&& fOnBodyPair) const
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.ForEachBodyPair");
    mBodiesBvh.SelfOverlaps(
        [this](IndexType oi, IndexType oj) {
            if (oi == oj)
            {
                throw std::runtime_error(
                    "MeshVertexTetrahedronDcd::UpdateActiveSet: oi == oj, which is not allowed.");
            }
            auto L1 = mBodyAabbs.col(oi).head<kDims>();
            auto U1 = mBodyAabbs.col(oi).tail<kDims>();
            auto L2 = mBodyAabbs.col(oj).head<kDims>();
            auto U2 = mBodyAabbs.col(oj).tail<kDims>();
            using math::linalg::mini::FromEigen;
            return geometry::OverlapQueries::AxisAlignedBoundingBoxes(
                FromEigen(L1),
                FromEigen(U1),
                FromEigen(L2),
                FromEigen(U2));
        },
        [fOnBodyPair = std::forward<FOnBodyPair>(
             fOnBodyPair)](IndexType oi, IndexType oj, [[maybe_unused]] IndexType k) {
            fOnBodyPair(oi, oj);
        });
}

template <class TDerivedX>
inline void MeshVertexTetrahedronDcd::MarkPenetratingVertices(
    IndexType ov,
    IndexType ot,
    Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.MarkPenetratingVertices");
    auto vBegin = mVP(ov);
    auto vEnd   = mVP(ov + 1);
    auto Xv     = X(Eigen::placeholders::all, mV(Eigen::seq(vBegin, vEnd - 1)));
    auto tBegin = mTP(ot);
    auto tEnd   = mTP(ot + 1);
    auto Lt =
        mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(tBegin, tEnd - 1)).topRows<kDims>();
    auto Ut = mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(tBegin, tEnd - 1))
                  .bottomRows<kDims>();
    std::unordered_map<IndexType, IndexType>& mBodyVertexToOtherBodiesValenceMap =
        mBodyVertexToOtherBodiesValence[static_cast<std::size_t>(ov)];
    mTetrahedronMeshHashGrids[static_cast<std::size_t>(ot)]
        .BroadPhase(Lt, Ut, Xv, [&](IndexType vq, IndexType tp) {
            auto const v  = vBegin + vq;
            auto const i  = mV(v);
            auto const XT = X(Eigen::placeholders::all, mT.col(tBegin + tp));
            using math::linalg::mini::FromEigen;
            if (geometry::OverlapQueries::PointTetrahedron3D(
                    FromEigen(X.col(i).head<kDims>()),
                    FromEigen(XT.col(0).head<kDims>()),
                    FromEigen(XT.col(1).head<kDims>()),
                    FromEigen(XT.col(2).head<kDims>()),
                    FromEigen(XT.col(3).head<kDims>())))
            {
                auto it  = mBodyVertexToOtherBodiesValenceMap.find(vq);
                auto end = mBodyVertexToOtherBodiesValenceMap.end();
                if (it == end)
                {
                    mBodyVertexToOtherBodiesValenceMap.insert({vq, IndexType(1)});
                }
                else
                {
                    auto& valence = it->second;
                    ++valence;
                }
            }
        });
}

template <class TDerivedX>
inline void MeshVertexTetrahedronDcd::FindNearestTrianglesToPenetratingVertices(
    IndexType ov,
    IndexType of,
    Eigen::DenseBase<TDerivedX> const& X)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.contact.MeshVertexTetrahedronDcd.FindNearestTrianglesToPenetratingVertices");
    auto vBegin = mVP(ov);
    auto vEnd   = mVP(ov + 1);
    auto Xv     = X(Eigen::placeholders::all, mV(Eigen::seq(vBegin, vEnd - 1)));
    auto fBegin = mFP(of);
    std::unordered_map<IndexType, IndexType>& mBodyVertexToOtherBodiesValenceMap =
        mBodyVertexToOtherBodiesValence[static_cast<std::size_t>(ov)];
    for (auto& [vq, valence] : mBodyVertexToOtherBodiesValenceMap)
    {
        assert(valence > 0);
        --valence;
        auto const v = vBegin + vq;
        auto const i = mV(v);
        auto const P = X.col(i).head<kDims>();
        using math::linalg::mini::FromEigen;
        mTriangleMeshBvhs[static_cast<std::size_t>(of)].NearestNeighbours(
            [&]<class TL, class TU>(TL const& L, TU const& U) {
                return geometry::DistanceQueries::PointAxisAlignedBoundingBox(
                    FromEigen(P),
                    FromEigen(L),
                    FromEigen(U));
            },
            [&](IndexType fq) { 
                auto const f = fBegin + fq;
                auto const iF = mF.col(f);
                auto const XT = X(Eigen::placeholders::all, iF);
                return geometry::DistanceQueries::PointTriangle(
                    FromEigen(P),
                    FromEigen(XT.col(0).head<kDims>()),
                    FromEigen(XT.col(1).head<kDims>()),
                    FromEigen(XT.col(2).head<kDims>()));
            },
            [&](IndexType fq, [[maybe_unused]] ScalarType d, IndexType k) {
                if (k >= kMaxVertexTriangleContacts)
                    return;
                mVFC(k, v) = fBegin + fq;
            }/*,
            radius,
            eps*/);
    }
    std::erase_if(mBodyVertexToOtherBodiesValenceMap, [](auto const& keyValuePair) {
        auto const& valence = keyValuePair.second;
        assert(valence >= 0);
        return valence == 0;
    });
}

} // namespace pbat::sim::contact

#endif // PBAT_SIM_CONTACT_MESHVERTEXTETRAHEDRONDCD_H