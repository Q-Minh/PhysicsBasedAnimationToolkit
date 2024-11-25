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
    ef = efIn;
    Nf = NfIn;
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
    bool const bShapeFunctionsAndElementsMatch = ef.size() == Nf.cols();
    if (not bShapeFunctionsAndElementsMatch)
    {
        std::string const what = fmt::format(
            "Expected ef.size() == Nf.cols(), but got dimensions ef={}, Nf={}x{}",
            ef.size(),
            Nf.rows(),
            Nf.cols());
        throw std::invalid_argument(what);
    }
    return *this;
}

void Prolongation::Apply(Hierarchy& H)
{
    auto lcStl                  = static_cast<std::size_t>(lc);
    auto lfStl                  = static_cast<std::size_t>(lf);
    Level const& Lc             = H.levels[lcStl];
    bool const bIsFineLevelRoot = lf < 0;
    MatrixX& xf                 = bIsFineLevelRoot ? H.root.x : H.levels[lfStl].C.x;
    assert(xf.cols() == Nf.cols() and xf.rows() == Lc.C.x.rows());
    tbb::parallel_for(Index(0), Nf.cols(), [&](Index i) {
        auto e    = ef(i);
        xf.col(i) = Lc.C.x(Eigen::placeholders::all, Lc.C.E.col(e)) * Nf.col(i);
    });
}

} // namespace vbd
} // namespace sim
} // namespace pbat