
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

template <int Dims>
class AxisAlignedBoundingBox : public Eigen::AlignedBox<Scalar, Dims>
{
  public:
    using BaseType = Eigen::AlignedBox<Scalar, Dims>;
    using SelfType = AxisAlignedBoundingBox;

    static auto constexpr kDims = Dims;

    AxisAlignedBoundingBox() = default;

    AxisAlignedBoundingBox(BaseType const& box);
    AxisAlignedBoundingBox(BaseType&& box);
    AxisAlignedBoundingBox& operator=(BaseType const& box);
    AxisAlignedBoundingBox& operator=(BaseType&& box);

    template <class TDerivedMin, class TDerivedMax>
    AxisAlignedBoundingBox(
        Eigen::DenseBase<TDerivedMin> const& min,
        Eigen::DenseBase<TDerivedMax> const& max);

    template <class TDerived>
    AxisAlignedBoundingBox(Eigen::DenseBase<TDerived> const& P);

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
