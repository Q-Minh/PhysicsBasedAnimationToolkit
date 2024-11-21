#include "Prolongation.h"

#include "Hierarchy.h"

#include <cassert>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

void Prolongation::Apply(Hierarchy& H)
{
    auto lcStl      = static_cast<std::size_t>(lc);
    Level const& Lc = H.levels[lcStl];
    if (lf > -1)
    {
        auto lfStl = static_cast<std::size_t>(lf);
        Level& Lf  = H.levels[lfStl];
        assert(Lf.C.x.cols() == Nf.cols() and Lf.C.x.rows() == Lc.C.x.rows());
        tbb::parallel_for(Index(0), Nf.cols(), [&](Index i) {
            auto e        = ef(i);
            Lf.C.x.col(i) = Lc.C.x(Eigen::placeholders::all, Lc.C.E.col(e)) * Nf.col(i);
        });
    }
    else
    {
        Data& Lf = H.root;
        assert(Lf.x.cols() == Nf.cols() and Lf.x.rows() == Lc.C.x.rows());
        tbb::parallel_for(Index(0), Nf.cols(), [&](Index i) {
            auto e      = ef(i);
            Lf.x.col(i) = Lc.C.x(Eigen::placeholders::all, Lc.C.E.col(e)) * Nf.col(i);
        });
    }
}

} // namespace vbd
} // namespace sim
} // namespace pbat