#ifndef PBAT_SIM_VBD_PROLONGATION_H
#define PBAT_SIM_VBD_PROLONGATION_H

#include "pbat/Aliases.h"

#include <cassert>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

class Prolongation
{
  public:
    /**
     * @brief Construct a new Prolongation object
     *
     * @tparam TDerivedTC
     * @tparam TDerivedEC
     * @tparam TDerivedN
     * @param TC
     * @param EC
     * @param N
     */
    template <class TDerivedTC, class TDerivedEC, class TDerivedN>
    Prolongation(
        Eigen::MatrixBase<TDerivedTC> const& TC,
        Eigen::MatrixBase<TDerivedEC> const& EC,
        Eigen::MatrixBase<TDerivedN> const& N);

    /**
     * @brief
     *
     * @tparam TDerivedXL
     * @tparam TDerivedXLM1
     * @param xl
     * @param xlm1
     */
    template <class TDerivedXL, class TDerivedXLM1>
    void Apply(Eigen::MatrixBase<TDerivedXL> const& xl, Eigen::MatrixBase<TDerivedXLM1>& xlm1);

  private:
    Eigen::Ref<IndexMatrixX const> TC; ///< Coarse (i.e. level l) tetrahedral mesh elements
    IndexVectorX EC; ///< Coarse (i.e. level l) elements associated with each fine level vertex
    MatrixX
        N; ///< 4x|V^{l-1}| linear tetrahedral shape functions at V^{l-1} (i.e. fine level vertices)
};

template <class TDerivedTC, class TDerivedEC, class TDerivedN>
inline Prolongation::Prolongation(
    Eigen::MatrixBase<TDerivedTC> const& TCin,
    Eigen::MatrixBase<TDerivedEC> const& ECin,
    Eigen::MatrixBase<TDerivedN> const& Nin)
    : TC(TCin), EC(ECin), N(Nin)
{
}

template <class TDerivedXL, class TDerivedXLM1>
inline void
Prolongation::Apply(Eigen::MatrixBase<TDerivedXL> const& xl, Eigen::MatrixBase<TDerivedXLM1>& xlm1)
{
    assert(xlm1.cols() == N.cols() and xlm1.rows() == xl.rows());
    tbb::parallel_for(Index(0), N.cols(), [&](Index i) {
        auto e      = EC(i);
        xlm1.col(i) = xl(Eigen::placeholders::all, TC.col(e)) * N.col(i);
    });
}

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_PROLONGATION_H