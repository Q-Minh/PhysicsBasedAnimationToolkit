#include "Prolongation.h"

#include "Hierarchy.h"

#include <cassert>
#include <exception>
#include <fmt/format.h>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

Prolongation& Prolongation::From(Index lcIn)
{
    lc = lcIn;
    return *this;
}

Prolongation& Prolongation::To(Index lfIn)
{
    lf = lfIn;
    return *this;
}

Prolongation& Prolongation::WithCoarseShapeFunctions(
    Eigen::Ref<IndexVectorX const> const& efIn,
    Eigen::Ref<MatrixX const> const& NfIn)
{
    ec = efIn;
    Nc = NfIn;
    return *this;
}

Prolongation& Prolongation::Construct(bool bValidate)
{
    if (not bValidate)
        return *this;

    bool const bTransitionValid = (lc > lf);
    if (not bTransitionValid)
    {
        std::string const what = fmt::format("Expected lc > lf, but got lc={}, lf={}", lc, lf);
        throw std::invalid_argument(what);
    }
    bool const bShapeFunctionsAndElementsMatch = ec.size() == Nc.cols();
    if (not bShapeFunctionsAndElementsMatch)
    {
        std::string const what = fmt::format(
            "Expected ef.size() == Nc.cols(), but got dimensions ef={}, Nc={}x{}",
            ec.size(),
            Nc.rows(),
            Nc.cols());
        throw std::invalid_argument(what);
    }
    return *this;
}

Index Prolongation::StartLevel() const
{
    return lc;
}

Index Prolongation::EndLevel() const
{
    return lf;
}

void Prolongation::Apply(Hierarchy& H)
{
    auto lcStl                  = static_cast<std::size_t>(lc);
    auto lfStl                  = static_cast<std::size_t>(lf);
    Level const& Lc             = H.levels[lcStl];
    bool const bIsFineLevelRoot = lf < 0;
    MatrixX& xf                 = bIsFineLevelRoot ? H.root.x : H.levels[lfStl].C.x;
    assert(xf.cols() == Nc.cols() and xf.rows() == Lc.C.x.rows());
    tbb::parallel_for(Index(0), Nc.cols(), [&](Index i) {
        auto e    = ec(i);
        xf.col(i) = Lc.C.x(Eigen::placeholders::all, Lc.C.E.col(e)) * Nc.col(i);
    });
}

} // namespace vbd
} // namespace sim
} // namespace pbat