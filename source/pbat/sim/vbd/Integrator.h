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

class PBAT_API Integrator
{
  public:
    Integrator(Data data);

    void Step(Scalar dt, Index iterations, Index substeps = Index{1});
    /**
     * @brief Mark next time step for optimization tracing.
     *
     * @param path Directory in which to save optimization traces.
     * @param t Time step index.
     */
    void TraceNextStep(std::string const& path = ".", Index t = -1);

    Data data;

    virtual ~Integrator() = default;

  protected:
    void InitializeSolve(Scalar sdt, Scalar sdt2);
    void RunVbdIteration(Scalar sdt, Scalar sdt2);
    virtual void Solve(Scalar sdt, Scalar sdt2, Index iterations);
    void SolveVertex(Index i, Scalar sdt, Scalar sdt2);

    /**
     * @brief The following methods are made public for debugging purposes (generally).
     */
  public:
    Scalar ObjectiveFunction(
        Eigen::Ref<MatrixX const> const& xk,
        Eigen::Ref<MatrixX const> const& xtilde,
        Scalar dt);
    VectorX ObjectiveFunctionGradient(
        Eigen::Ref<MatrixX const> const& xk,
        Eigen::Ref<MatrixX const> const& xtilde,
        Scalar dt);

  protected:
    void ExportTrace(Scalar sdt, Index substep);
    void TryTraceIteration(Scalar sdt);

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
