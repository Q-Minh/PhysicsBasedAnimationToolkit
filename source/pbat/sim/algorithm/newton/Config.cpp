#include "Config.h"

#include "pbat/io/Archive.h"

#include <exception>
#include <fmt/format.h>

namespace pbat::sim::algorithm::newton {

Config& Config::WithSubsteps(int substeps)
{
    nSubsteps = substeps;
    return *this;
}

Config& Config::WithConvergence(
    int maxAugmentedLagrangianIterations,
    int maxNewtonIterations,
    Scalar gtolIn,
    int maxLinearSolverIterations,
    Scalar rtolIn)
{
    nMaxAugmentedLagrangianIterations = maxAugmentedLagrangianIterations;
    nMaxNewtonIterations              = maxNewtonIterations;
    gtol                              = gtolIn;
    nMaxLinearSolverIterations        = maxLinearSolverIterations;
    rtol                              = rtolIn;
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
    if (nMaxAugmentedLagrangianIterations < 1)
    {
        throw std::invalid_argument(
            fmt::format(
                "0 < maxAugmentedLagrangianIterations, but got maxAugmentedLagrangianIterations={}",
                nMaxAugmentedLagrangianIterations));
    }
    if (nMaxNewtonIterations < 1)
    {
        throw std::invalid_argument(
            fmt::format("0 < maxIterations, but got maxIterations={}", nMaxNewtonIterations));
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

void Config::Serialize(io::Archive& archive) const
{
    io::Archive group = archive["pbat.sim.algorithm.newton.Config"];
    group.WriteData("nSubsteps", nSubsteps);
    group.WriteData("nMaxAugmentedLagrangianIterations", nMaxAugmentedLagrangianIterations);
    group.WriteData("nMaxNewtonIterations", nMaxNewtonIterations);
    group.WriteData("gtol", gtol);
    group.WriteData("nMaxLinearSolverIterations", nMaxLinearSolverIterations);
    group.WriteData("rtol", rtol);
    group.WriteData("nMaxLineSearchIterations", nMaxLineSearchIterations);
    group.WriteData("tauArmijo", tauArmijo);
    group.WriteData("cArmijo", cArmijo);
    group.WriteData("muC", muC);
}

void Config::Deserialize(io::Archive const& archive)
{
    io::Archive const group           = archive["pbat.sim.algorithm.newton.Config"];
    nSubsteps                         = group.ReadData<decltype(nSubsteps)>("nSubsteps");
    nMaxAugmentedLagrangianIterations = group.ReadData<decltype(nMaxAugmentedLagrangianIterations)>(
        "nMaxAugmentedLagrangianIterations");
    nMaxNewtonIterations = group.ReadData<decltype(nMaxNewtonIterations)>("nMaxNewtonIterations");
    gtol                 = group.ReadData<decltype(gtol)>("gtol");
    nMaxLinearSolverIterations =
        group.ReadData<decltype(nMaxLinearSolverIterations)>("nMaxLinearSolverIterations");
    rtol = group.ReadData<decltype(rtol)>("rtol");
    nMaxLineSearchIterations =
        group.ReadData<decltype(nMaxLineSearchIterations)>("nMaxLineSearchIterations");
    tauArmijo = group.ReadData<decltype(tauArmijo)>("tauArmijo");
    cArmijo   = group.ReadData<decltype(cArmijo)>("cArmijo");
    muC       = group.ReadData<decltype(muC)>("muC");
}

} // namespace pbat::sim::algorithm::newton