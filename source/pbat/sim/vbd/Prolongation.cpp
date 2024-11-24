#include "Prolongation.h"

#include "Hierarchy.h"

#include <cassert>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

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