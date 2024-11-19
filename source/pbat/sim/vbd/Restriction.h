#ifndef PBAT_SIM_VBD_RESTRICTION_H
#define PBAT_SIM_VBD_RESTRICTION_H

#include "pbat/Aliases.h"

namespace pbat {
namespace sim {
namespace vbd {

class Restriction
{
  public:
    /**
     * @brief
     *
     * @tparam TDerivedXL
     * @tparam TDerivedXLP1
     * @param xl
     * @param xlp1
     */
    template <class TDerivedXL, class TDerivedXLP1>
    void Apply(Eigen::MatrixBase<TDerivedXL> const& xl, Eigen::MatrixBase<TDerivedXLP1>& xlp1);

  private:
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_RESTRICTION_H