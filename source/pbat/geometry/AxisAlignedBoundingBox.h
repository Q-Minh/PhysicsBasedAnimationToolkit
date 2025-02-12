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

namespace pbat {
namespace geometry {

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

} // namespace geometry
} // namespace pbat

#endif // PBAT_GEOMETRY_AXISALIGNEDBOUNDINGBOX_H
