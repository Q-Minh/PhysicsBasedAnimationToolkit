/**
 * @file AxisAlignedBoundingBox.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Axis-aligned bounding box class
 * @version 0.1
 * @date 2025-02-12
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef PBAT_GEOMETRY_AXISALIGNEDBOUNDINGBOX_H
#define PBAT_GEOMETRY_AXISALIGNEDBOUNDINGBOX_H

#include <Eigen/Geometry>
#include <array>
#include <exception>
#include <fmt/format.h>
#include <pbat/Aliases.h>
#include <string>
#include <vector>

namespace pbat::geometry {

/**
 * @brief Axis-aligned bounding box class
 *
 * @note This class is a thin wrapper around Eigen::AlignedBox
 *
 * @tparam Dims Number of dimensions
 */
template <int Dims>
class AxisAlignedBoundingBox : public Eigen::AlignedBox<Scalar, Dims>
{
  public:
    using BaseType = Eigen::AlignedBox<Scalar, Dims>; ///< Base type
    using SelfType = AxisAlignedBoundingBox;          ///< Self type

    static auto constexpr kDims = Dims; ///< Number of dimensions

    AxisAlignedBoundingBox() = default;

    /**
     * @brief Copy construct AxisAlignedBoundingBox from Eigen::AlignedBox
     * @param box Eigen::AlignedBox
     */
    AxisAlignedBoundingBox(BaseType const& box);
    /**
     * @brief Move construct AxisAlignedBoundingBox from Eigen::AlignedBox
     * @param box Eigen::AlignedBox
     */
    AxisAlignedBoundingBox(BaseType&& box);
    /**
     * @brief Copy assign AxisAlignedBoundingBox from Eigen::AlignedBox
     * @param box Eigen::AlignedBox
     * @return Reference to this
     */
    AxisAlignedBoundingBox& operator=(BaseType const& box);
    /**
     * @brief Move assign AxisAlignedBoundingBox from Eigen::AlignedBox
     * @param box Eigen::AlignedBox
     * @return Reference to this
     */
    AxisAlignedBoundingBox& operator=(BaseType&& box);
    /**
     * @brief Construct AxisAlignedBoundingBox from min and max endpoints
     * @tparam TDerivedMin Eigen dense expression type
     * @tparam TDerivedMax Eigen dense expression type
     * @param min Min endpoint
     * @param max Max endpoint
     * @pre `min.rows() == Dims` and `max.rows() == Dims`
     */
    template <class TDerivedMin, class TDerivedMax>
    AxisAlignedBoundingBox(
        Eigen::DenseBase<TDerivedMin> const& min,
        Eigen::DenseBase<TDerivedMax> const& max);
    /**
     * @brief Construct AxisAlignedBoundingBox over a set of points
     * @tparam TDerived Eigen dense expression type
     * @param P Points
     * @pre `P.rows() == Dims`
     */
    template <class TDerived>
    AxisAlignedBoundingBox(Eigen::DenseBase<TDerived> const& P);
    /**
     * @brief Get indices of points in P contained in the bounding box
     * @tparam TDerived Eigen dense expression type
     * @param P Points
     * @return Indices of points in P contained in the bounding box
     */
    template <class TDerived>
    std::vector<Index> contained(Eigen::MatrixBase<TDerived> const& P) const;
};

template <int Dims>
inline AxisAlignedBoundingBox<Dims>::AxisAlignedBoundingBox(BaseType const& box) : BaseType(box)
{
}

template <int Dims>
inline AxisAlignedBoundingBox<Dims>::AxisAlignedBoundingBox(BaseType&& box) : BaseType(box)
{
}

template <int Dims>
inline AxisAlignedBoundingBox<Dims>& AxisAlignedBoundingBox<Dims>::operator=(BaseType const& box)
{
    BaseType::template operator=(box);
    return *this;
}

template <int Dims>
inline AxisAlignedBoundingBox<Dims>& AxisAlignedBoundingBox<Dims>::operator=(BaseType&& box)
{
    BaseType::template operator=(box);
    return *this;
}

template <int Dims>
template <class TDerivedMin, class TDerivedMax>
inline AxisAlignedBoundingBox<Dims>::AxisAlignedBoundingBox(
    Eigen::DenseBase<TDerivedMin> const& min,
    Eigen::DenseBase<TDerivedMax> const& max)
    : BaseType(min, max)
{
}

template <int Dims>
template <class TDerived>
inline AxisAlignedBoundingBox<Dims>::AxisAlignedBoundingBox(Eigen::DenseBase<TDerived> const& P)
{
    if (P.rows() != Dims)
    {
        std::string const what = fmt::format(
            "Expected points P of dimensions {}x|#points|, but got {}x{}",
            Dims,
            P.rows(),
            P.cols());
        throw std::invalid_argument(what);
    }
    for (auto i = 0; i < P.cols(); ++i)
    {
        BaseType::template extend(P.col(i));
    }
}

template <int Dims>
template <class TDerived>
inline std::vector<Index>
AxisAlignedBoundingBox<Dims>::contained(Eigen::MatrixBase<TDerived> const& P) const
{
    std::vector<Index> inds{};
    inds.reserve(static_cast<std::size_t>(P.cols()));
    for (auto i = 0; i < P.cols(); ++i)
    {
        if (BaseType::template contains(P.col(i)))
        {
            inds.push_back(static_cast<Index>(i));
        }
    }
    return inds;
}

/**
 * @brief Computes AABBs of nClusters kDims-dimensional point clusters
 *
 * @tparam kDims Spatial dimension
 * @tparam kClusterNodes Number of nodes in a cluster
 * @tparam FCluster Function with signature `auto (Index) -> std::convertible_to<Matrix<kDims,
 * kClusterNodes>>`
 * @tparam TDerivedB Eigen dense expression type
 * @param fCluster Function to get the cluster at index `c`
 * @param nClusters Number of clusters
 * @param B 2*kDims x |# clusters| output AABBs
 */
template <auto kDims, auto kClusterNodes, class FCluster, class TDerivedB>
inline void ClustersToAabbs(FCluster fCluster, Index nClusters, Eigen::DenseBase<TDerivedB>& B)
{
    for (auto c = 0; c < nClusters; ++c)
    {
        Matrix<kDims, kClusterNodes> XC = fCluster(c);
        B.col(c).head<kDims>()          = XC.rowwise().minCoeff();
        B.col(c).tail<kDims>()          = XC.rowwise().maxCoeff();
    }
}

/**
 * @brief Computes AABBs of nElemNodes simplex mesh elements in kDims dimensions
 *
 * @tparam kDims Spatial dimension
 * @tparam kElemNodes Number of nodes in an element
 * @tparam TDerivedX Eigen dense expression type
 * @tparam TDerivedE Eigen dense expression type
 * @tparam TDerivedB Eigen dense expression type
 * @param X `kDims x |# nodes|` matrix of node positions
 * @param E `kElemNodes x |# elements|` matrix of element node indices
 * @param B 2*kDims x |# elements| output AABBs
 */
template <auto kDims, auto kElemNodes, class TDerivedX, class TDerivedE, class TDerivedB>
inline void MeshToAabbs(
    Eigen::DenseBase<TDerivedX> const& X,
    Eigen::DenseBase<TDerivedE> const& E,
    Eigen::DenseBase<TDerivedB>& B)
{
    ClustersToAabbs<kDims, kElemNodes>(
        [&](Index e) {
            return X(Eigen::placeholders::all, E.col(e)).block<kDims, kElemNodes>(0, 0);
        },
        E.cols(),
        B);
}

} // namespace pbat::geometry

#endif // PBAT_GEOMETRY_AXISALIGNEDBOUNDINGBOX_H
