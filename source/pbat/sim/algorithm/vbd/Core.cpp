#include "Core.h"

#include <exception>
#include <fmt/core.h>

namespace pbat::sim::algorithm::vbd {

PBAT_API Params& Params::WithVertexElementAdjacencyGraph(
    Eigen::Ref<IndexVectorX const> const& _GVGp,
    Eigen::Ref<IndexVectorX const> const& _GVGe,
    Eigen::Ref<IndexVectorX const> const& _GVGilocal)
{
    GVGp      = _GVGp;
    GVGe      = _GVGe;
    GVGilocal = _GVGilocal;
    return *this;
}

PBAT_API Params& Params::WithVertexColors(Eigen::Ref<IndexVectorX const> const& _colors)
{
    colors = _colors;
    return *this;
}

PBAT_API Params& Params::WithInitializationStrategy(EInitializationStrategy _strategy)
{
    strategy = _strategy;
    return *this;
}

PBAT_API Params& Params::WithHessianDeterminantZeroUnder(Scalar zero)
{
    detHZero = zero;
    return *this;
}

PBAT_API Params& Params::Construct(bool bValidate)
{
    if (bValidate)
    {
        auto nVerts = colors.size();
        if (GVGp.size() != nVerts + 1)
        {
            throw std::invalid_argument(
                fmt::format(
                    "GVGp size {} inconsistent with expected # verts {}",
                    GVGp.size(),
                    nVerts));
        }
        if (Padj.size() != nVerts)
        {
            throw std::invalid_argument(
                fmt::format(
                    "Padj size {} inconsistent with expected # verts {}",
                    Padj.size(),
                    nVerts));
        }
        auto nPartitions = Pptr.size() - 1;
        if (colors.maxCoeff() + 1 != nPartitions)
        {
            throw std::invalid_argument(
                fmt::format(
                    "# colors {} inconsistent with expected # partitions {}",
                    colors.maxCoeff() + 1,
                    nPartitions));
        }
        auto nVertexElementAdjacencies = GVGe.size();
        if (GVGilocal.size() != nVertexElementAdjacencies)
        {
            throw std::invalid_argument(
                fmt::format(
                    "GVGilocal size {} inconsistent with expected # vertex-element adjacencies {}",
                    GVGilocal.size(),
                    nVertexElementAdjacencies));
        }
    }
    return *this;
}

} // namespace pbat::sim::algorithm::vbd