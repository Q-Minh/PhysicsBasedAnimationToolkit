#ifndef PBAT_SIM_VBD_INTEGRATOR_H
#define PBAT_SIM_VBD_INTEGRATOR_H

#include "Data.h"
#include "PhysicsBasedAnimationToolkitExport.h"
#include "pbat/Aliases.h"

#include <string>
#include <vector>

namespace pbat {
namespace sim {
namespace vbd {

class Integrator
{
  public:
    PBAT_API Integrator(Data data);

    PBAT_API void Step(Scalar dt, Index iterations, Index substeps = Index{1});

    Data data;

    PBAT_API virtual ~Integrator() = default;

  protected:
    PBAT_API void InitializeSolve(Scalar sdt, Scalar sdt2);
    PBAT_API void RunVbdIteration(Scalar sdt, Scalar sdt2);
    PBAT_API virtual void Solve(Scalar sdt, Scalar sdt2, Index iterations);
    PBAT_API void SolveVertex(Index i, Scalar sdt, Scalar sdt2);

    /**
     * @brief The following methods are made public for debugging purposes (generally).
     */
  public:
    PBAT_API Scalar ObjectiveFunction(
        Eigen::Ref<MatrixX const> const& xk,
        Eigen::Ref<MatrixX const> const& xtilde,
        Scalar dt);
    PBAT_API VectorX ObjectiveFunctionGradient(
        Eigen::Ref<MatrixX const> const& xk,
        Eigen::Ref<MatrixX const> const& xtilde,
        Scalar dt);
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_INTEGRATOR_H
