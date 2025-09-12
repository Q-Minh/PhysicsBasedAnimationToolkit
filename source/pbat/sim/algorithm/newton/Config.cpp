#include "Config.h"

#include <exception>
#include <fmt/format.h>

namespace pbat::sim::algorithm::newton {

Config& Config::WithSubsteps(int substeps)
{
    nSubsteps = substeps;
    return *this;
}

Config& Config::WithConvergence(
    int maxIterations,
    Scalar gtolIn,
    int maxLinearSolverIterations,
    Scalar rtolIn)
{
    nMaxIterations             = maxIterations;
    gtol                       = gtolIn;
    nMaxLinearSolverIterations = maxLinearSolverIterations;
    rtol                       = rtolIn;
    return *this;
}

Config& Config::WithLineSearch(int maxLineSearchIterations, Scalar tauArmijoIn, Scalar cArmijoIn)
{
    nMaxLineSearchIterations = maxLineSearchIterations;
    tauArmijo                = tauArmijoIn;
    cArmijo                  = cArmijoIn;
    return *this;
}

Config& Config::WithContactParameters(Scalar muCIn)
{
    muC = muCIn;
    return *this;
}

Config& Config::Construct()
{
    if (nSubsteps < 1)
    {
        throw std::invalid_argument(fmt::format("0 < substeps, but got substeps={}", nSubsteps));
    }
    if (nMaxIterations < 1)
    {
        throw std::invalid_argument(
            fmt::format("0 < maxIterations, but got maxIterations={}", nMaxIterations));
    }
    if (nMaxLinearSolverIterations < 1)
    {
        throw std::invalid_argument(
            fmt::format(
                "0 < maxLinearSolverIterations, but got maxLinearSolverIterations={}",
                nMaxLinearSolverIterations));
    }
    if (nMaxLineSearchIterations < 1)
    {
        throw std::invalid_argument(
            fmt::format(
                "0 < maxLineSearchIterations, but got maxLineSearchIterations={}",
                nMaxLineSearchIterations));
    }
    if (tauArmijo <= Scalar(0) or tauArmijo >= Scalar(1))
    {
        throw std::invalid_argument(
            fmt::format("0 < tauArmijo < 1, but got tauArmijo={}", tauArmijo));
    }
    if (cArmijo <= Scalar(0) or cArmijo >= Scalar(1))
    {
        throw std::invalid_argument(fmt::format("0 < cArmijo < 1, but got cArmijo={}", cArmijo));
    }
    if (muC <= Scalar(0))
    {
        throw std::invalid_argument(fmt::format("0 < muC, but got muC={}", muC));
    }
    return *this;
}

} // namespace pbat::sim::algorithm::newton