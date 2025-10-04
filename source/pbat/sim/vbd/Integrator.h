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
    /**
     * @brief Mark next time step for optimization tracing.
     *
     * @param path Directory in which to save optimization traces.
     * @param t Time step index.
     */
    PBAT_API void TraceNextStep(std::string const& path = ".", Index t = -1);

    Data data;

    virtual ~Integrator() = default;

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

  protected:
    PBAT_API void ExportTrace(Scalar sdt, Index substep);
    PBAT_API void TryTraceIteration(Scalar sdt);

  private:
    bool mTraceIterates{false};
    std::string mTracePath{"."};
    Index mTimeStep{-1};
    std::vector<Scalar> mTracedObjectives;
    std::vector<VectorX> mTracedGradients;
    std::vector<MatrixX> mTracedPositions;
};

} // namespace vbd
} // namespace sim
} // namespace pbat

#endif // PBAT_SIM_VBD_INTEGRATOR_H
