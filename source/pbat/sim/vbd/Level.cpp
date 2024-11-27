#include "Level.h"

#include "Kernels.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/physics/StableNeoHookeanEnergy.h"

#include <exception>
#include <fmt/format.h>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {

Level::Energy& Level::Energy::WithQuadrature(
    Eigen::Ref<VectorX const> const& wgIn,
    Eigen::Ref<Eigen::Vector<bool, Eigen::Dynamic> const> const& sgIn)
{
    wg = wgIn;
    sg = sgIn;
    return *this;
}

Level::Energy& Level::Energy::WithAdjacency(
    Eigen::Ref<IndexVectorX const> const& GVGpIn,
    Eigen::Ref<IndexVectorX const> const& GVGgIn,
    Eigen::Ref<IndexVectorX const> const& GVGeIn,
    Eigen::Ref<IndexVectorX const> const& GVGilocalIn)
{
    GVGp      = GVGpIn;
    GVGg      = GVGgIn;
    GVGe      = GVGeIn;
    GVGilocal = GVGilocalIn;
    return *this;
}

Level::Energy& Level::Energy::WithKineticEnergy(
    Eigen::Ref<VectorX const> const& rhogIn,
    Eigen::Ref<MatrixX const> const& NcgIn)
{
    rhog = rhogIn;
    Ncg  = NcgIn;
    xtildeg.resize(3, rhog.size());
    return *this;
}

Level::Energy& Level::Energy::WithPotentialEnergy(
    Eigen::Ref<VectorX const> const& mugIn,
    Eigen::Ref<VectorX const> const& lambdagIn,
    Eigen::Ref<IndexVectorX const> const& ergIn,
    Eigen::Ref<MatrixX const> const& NrgIn,
    Eigen::Ref<MatrixX const> const& GNfgIn,
    Eigen::Ref<MatrixX const> const& GNcgIn)
{
    mug     = mugIn;
    lambdag = lambdagIn;
    erg     = ergIn;
    Nrg     = NrgIn;
    GNfg    = GNfgIn;
    GNcg    = GNcgIn;
    return *this;
}

Level::Energy& Level::Energy::Construct(bool bValidate)
{
    if (not bValidate)
        return *this;

    auto const nQuadPts = wg.size();
    bool const bQuadratureValid =
        (xtildeg.rows() == 3 and xtildeg.cols() == nQuadPts) and (rhog.size() == nQuadPts) and
        (Ncg.rows() == 4 and Ncg.cols() == nQuadPts) and (mug.size() == nQuadPts) and
        (lambdag.size() == nQuadPts) and (erg.rows() == 4 and erg.cols() == nQuadPts) and
        (Nrg.rows() == 4 and Nrg.cols() == 4 * nQuadPts) and
        (GNfg.rows() == 4 and GNfg.cols() == 3 * nQuadPts) and
        (GNcg.rows() == 4 and GNcg.cols() == 3 * nQuadPts) and (sg.size() == nQuadPts);
    if (not bQuadratureValid)
    {
        std::string const what = fmt::format(
            "With #quad.pts.={0}, expected dimensions xtilde=3x{0}, rhog={0}, Ncg=4x{0}, mug={0}, "
            "lambdag={0}, erg=4x{0}, Nrg=4x{1}, GNfg=4x{2}, GNcg=4x{2}, sg={0}",
            nQuadPts,
            4 * nQuadPts,
            3 * nQuadPts);
        throw std::invalid_argument(what);
    }
    bool const bAdjacencyValid = (GVGg.size() == GVGe.size()) and (GVGg.size() == GVGilocal.size());
    if (not bAdjacencyValid)
    {
        std::string const what = fmt::format(
            "With the vertex-quad.pt. adjacency graph having #edges={0}, expected dimensions "
            "GVGe={0}, GVGilocal={0}",
            GVGg.size());
        throw std::invalid_argument(what);
    }
    return *this;
}

Level::Cage::Cage(
    Eigen::Ref<MatrixX const> const& xIn,
    Eigen::Ref<IndexMatrixX const> const& Ein,
    Eigen::Ref<IndexVectorX const> const& ptrIn,
    Eigen::Ref<IndexVectorX const> const& adjIn)
    : x(xIn), E(Ein), ptr(ptrIn), adj(adjIn)
{
}

Level::RootParameterBus::RootParameterBus(
    Eigen::Ref<IndexVectorX const> const& ergIn,
    Eigen::Ref<MatrixX const> const& NrgIn)
    : erg(ergIn), Nrg(NrgIn)
{
}

Level::Level(Cage Cin, Energy Ein, RootParameterBus RPBin)
    : C(std::move(Cin)), E(std::move(Ein)), RPB(std::move(RPBin))
{
}

} // namespace vbd
} // namespace sim
} // namespace pbat