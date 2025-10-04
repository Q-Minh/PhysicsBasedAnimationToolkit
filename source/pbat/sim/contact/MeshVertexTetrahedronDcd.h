#ifndef PBAT_SIM_CONTACT_MESHVERTEXTETRAHEDRONDCD_H
#define PBAT_SIM_CONTACT_MESHVERTEXTETRAHEDRONDCD_H

#include "MultibodyTetrahedralMeshSystem.h"
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

/**
 * @brief Discrete contact detection system for vertex-tetrahedron contacts between multiple
 * tetrahedral meshes.
 *
 * Uses an AABB tree hierarchy for the mesh vertices and an AABB tree hierarchy for the mesh
 * triangles, as well as a spatial hash grid for the tetrahedra.
 *
 * For each vertex, stores up to `kMaxVertexTriangleContacts` contacting triangles.
 */
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
     * @param T `4 x |# tetrahedra|` tetrahedron array
     */
    MeshVertexTetrahedronDcd(
        Eigen::Ref<Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic>> X,
        Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic>> T);
    /**
     * @brief Update the active set of vertex-triangle contacts
     * @tparam TDerivedX Eigen type of vertex positions
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param T `4 x |# tetrahedra|` tetrahedron array
     */
    void UpdateActiveSet(
        Eigen::Ref<Eigen::Matrix<ScalarType, kDims, Eigen::Dynamic> const> const& X,
        Eigen::Ref<Eigen::Matrix<IndexType, 4, Eigen::Dynamic> const> const& T);
    /**
     * @brief Visit the triangles in active vertex-triangle contacts
     * @tparam FOnVertexTriangleContactPair Callable type with signature `void(
     * IndexType f, IndexType k)` where `f` is the index of the triangle and `0 <= k <
     * kMaxVertexTriangleContacts` is the index of the contact
     * @param v Index of the vertex whose active triangle contacts to visit
     * @param fOnVertexTriangleContactPair Function to call for each vertex-triangle contact pair
     */
    template <class FOnVertexTriangleContactPair>
    void ForEachActiveVertexTriangleContact(
        IndexType v,
        FOnVertexTriangleContactPair fOnVertexTriangleContactPair) const;
    /**
     * @brief Get the multibody tetrahedral mesh system
     */
    MultibodyTetrahedralMeshSystem<IndexType> const& MultibodySystem() const;

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
     * @tparam TDerivedT Eigen type of tetrahedron mesh elements/connectivity
     * @param o Index of the body to compute tetrahedron AABBs for
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param T `4 x |# tetrahedra|` tetrahedron mesh elements/connectivity
     */
    template <class TDerivedX, class TDerivedT>
    void ComputeTetrahedronMeshAabbs(
        IndexType o,
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::DenseBase<TDerivedT> const& T);
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
     * @tparam TDerivedT Eigen type of tetrahedron mesh elements/connectivity
     * @param ov Index of the body whose collision vertices to check against body `ot`
     * @param ot Index of the body penetrated by body `ov`
     * @param X `kDims x |# vertices|` matrix of vertex positions
     * @param T `4 x |# tetrahedra|` tetrahedron mesh elements/connectivity
     */
    template <class TDerivedX, class TDerivedT>
    void MarkPenetratingVertices(
        IndexType ov,
        IndexType ot,
        Eigen::DenseBase<TDerivedX> const& X,
        Eigen::DenseBase<TDerivedT> const& T);
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
    sim::contact::MultibodyTetrahedralMeshSystem<IndexType>
        mMultibodySystem; ///< Multibody system containing the tetrahedral mesh bodies.

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
    auto const nTriangles = mMultibodySystem.NumContactTriangles();
    for (auto f = 0; f < nTriangles; ++f)
    {
        Matrix<kDims, 3> XF;
        XF = X(Eigen::placeholders::all, mMultibodySystem.F.col(f)).template block<kDims, 3>(0, 0);
        auto L = mTriangleAabbs.col(f).head<kDims>();
        auto U = mTriangleAabbs.col(f).tail<kDims>();
        L      = XF.rowwise().minCoeff();
        U      = XF.rowwise().maxCoeff();
    }
}

template <class TDerivedX, class TDerivedT>
inline void MeshVertexTetrahedronDcd::ComputeTetrahedronMeshAabbs(
    IndexType o,
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedT> const& T)
{
    PBAT_PROFILE_NAMED_SCOPE(
        "pbat.sim.contact.MeshVertexTetrahedronDcd.ComputeTetrahedronMeshAabbs");
    auto const [tbegin, tend] = mMultibodySystem.TetrahedraRangeFor(o);
    auto TB = mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(tbegin, tend - 1));
    for (auto t = tbegin; t < tend; ++t)
    {
        Matrix<kDims, 4> XT = X(Eigen::placeholders::all, T.col(t)).template block<kDims, 4>(0, 0);
        auto L              = TB.col(t - tbegin).template head<kDims>();
        auto U              = TB.col(t - tbegin).template tail<kDims>();
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
            auto L1 = mBodyAabbs.col(oi).template head<kDims>();
            auto U1 = mBodyAabbs.col(oi).template tail<kDims>();
            auto L2 = mBodyAabbs.col(oj).template head<kDims>();
            auto U2 = mBodyAabbs.col(oj).template tail<kDims>();
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

template <class TDerivedX, class TDerivedT>
inline void MeshVertexTetrahedronDcd::MarkPenetratingVertices(
    IndexType ov,
    IndexType ot,
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedT> const& T)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.contact.MeshVertexTetrahedronDcd.MarkPenetratingVertices");
    auto const Vov            = mMultibodySystem.ContactVerticesOf(ov);
    auto Xv                   = X(Eigen::placeholders::all, Vov);
    auto const [tbegin, tend] = mMultibodySystem.TetrahedraRangeFor(ot);
    auto Lt =
        mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(tbegin, tend - 1)).topRows<kDims>();
    auto Ut = mTetrahedronAabbs(Eigen::placeholders::all, Eigen::seq(tbegin, tend - 1))
                  .bottomRows<kDims>();
    std::unordered_map<IndexType, IndexType>& mBodyVertexToOtherBodiesValenceMap =
        mBodyVertexToOtherBodiesValence[static_cast<std::size_t>(ov)];
    mTetrahedronMeshHashGrids[static_cast<std::size_t>(ot)]
        .BroadPhase(Lt, Ut, Xv, [&](IndexType vq, IndexType tp) {
            auto const i  = Vov(vq);
            auto const XT = X(Eigen::placeholders::all, T.col(tbegin + tp));
            using math::linalg::mini::FromEigen;
            if (geometry::OverlapQueries::PointTetrahedron3D(
                    FromEigen(X.col(i).template head<kDims>()),
                    FromEigen(XT.col(0).template head<kDims>()),
                    FromEigen(XT.col(1).template head<kDims>()),
                    FromEigen(XT.col(2).template head<kDims>()),
                    FromEigen(XT.col(3).template head<kDims>())))
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
    IndexType const vbegin = mMultibodySystem.ContactVerticesRangeFor(ov).first;
    auto const fbegin      = mMultibodySystem.ContactTrianglesRangeFor(of).first;
    std::unordered_map<IndexType, IndexType>& mBodyVertexToOtherBodiesValenceMap =
        mBodyVertexToOtherBodiesValence[static_cast<std::size_t>(ov)];
    for (auto& [vq, valence] : mBodyVertexToOtherBodiesValenceMap)
    {
        assert(valence > 0);
        --valence;
        auto const v = vbegin + vq;
        auto const i = mMultibodySystem.V(v);
        auto const P = X.col(i).template head<kDims>();
        using math::linalg::mini::FromEigen;
        mTriangleMeshBvhs[static_cast<std::size_t>(of)].NearestNeighbours(
            [&]<class TL, class TU>(TL const& L, TU const& U) {
                return geometry::DistanceQueries::PointAxisAlignedBoundingBox(
                    FromEigen(P),
                    FromEigen(L),
                    FromEigen(U));
            },
            [&](IndexType fq) { 
                auto const f = fbegin + fq;
                auto const iF = mMultibodySystem.F.col(f);
                auto const XT = X(Eigen::placeholders::all, iF);
                return geometry::DistanceQueries::PointTriangle(
                    FromEigen(P),
                    FromEigen(XT.col(0).template head<kDims>()),
                    FromEigen(XT.col(1).template head<kDims>()),
                    FromEigen(XT.col(2).template head<kDims>()));
            },
            [&](IndexType fq, [[maybe_unused]] ScalarType d, IndexType k) {
                if (k >= kMaxVertexTriangleContacts)
                    return;
                mVFC(k, v) = fbegin + fq;
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