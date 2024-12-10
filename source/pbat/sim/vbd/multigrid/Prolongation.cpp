#include "Prolongation.h"

#include "Level.h"
#include "pbat/fem/Jacobian.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/geometry/TetrahedralAabbHierarchy.h"

#include <exception>
#include <fmt/format.h>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace vbd {
namespace multigrid {

Prolongation::Prolongation(VolumeMesh const& FM, VolumeMesh const& CM) : ec(), Nc()
{
    geometry::TetrahedralAabbHierarchy cbvh(CM.X, CM.E);
    ec = cbvh.PrimitivesContainingPoints(FM.X);
    if ((ec.array() == Index(-1)).any())
    {
        auto const nOutOfBounds = (ec.array() == Index(-1)).count();
        std::string const what  = fmt::format(
            "Expected mesh CM to embed mesh FM, but found {} vertices of FM outside of CM",
            nOutOfBounds);
        throw std::invalid_argument(what);
    }
    Nc = fem::ShapeFunctionsAt(CM, ec, FM.X);
}

void Prolongation::Apply(Level const& lc, Level& lf)
{
    VolumeMesh const& CM = lc.mesh;
    MatrixX const& xc    = lc.x;
    MatrixX& xf          = lf.x;
    tbb::parallel_for(Index(0), ec.size(), [&](Index i) {
        auto e    = ec(i);
        auto inds = CM.E(Eigen::placeholders::all, e);
        auto N    = Nc.col(i);
        xf.col(i) = xc(Eigen::placeholders::all, inds) * N;
    });
}

} // namespace multigrid
} // namespace vbd
} // namespace sim
} // namespace pbat

#include "pbat/geometry/model/Cube.h"
#ifdef PBAT_WITH_PRECOMPILED_LARGE_MODELS
    #include "pbat/geometry/model/Armadillo.h"
#endif // PBAT_WITH_PRECOMPILED_LARGE_MODELS

#include <doctest/doctest.h>

TEST_CASE("[sim][vbd][multigrid] Prolongation")
{
    using namespace pbat;
    using sim::vbd::multigrid::Prolongation;
    using sim::vbd::multigrid::VolumeMesh;

    auto const fActAndAssert = [](VolumeMesh const& FM, VolumeMesh const& CM) {
        Prolongation P(FM, CM);
        CHECK_EQ(P.Nc.cols(), FM.X.cols());
        CHECK_EQ(P.ec.size(), FM.X.cols());
    };

    SUBCASE("Cube")
    {
        auto [VR, CR] = geometry::model::Cube();
        // Center and create cage
        VR.colwise() -= VR.rowwise().mean();
        MatrixX VC      = Scalar(1.1) * VR;
        IndexMatrixX CC = CR;
        VolumeMesh FM(VR, CR);
        VolumeMesh CM(VC, CC);
        fActAndAssert(FM, CM);
    }
#ifdef PBAT_WITH_PRECOMPILED_LARGE_MODELS
    SUBCASE("Armadillo")
    {
        auto [VR, CR] = geometry::model::Armadillo(geometry::model::EMesh::Tetrahedral, Index(0));
        auto [VC, CC] = geometry::model::Armadillo(geometry::model::EMesh::Tetrahedral, Index(1));
        VolumeMesh FM(VR, CR);
        VolumeMesh CM(VC, CC);
        fActAndAssert(FM, CM);
    }
#endif // PBAT_WITH_PRECOMPILED_LARGE_MODELS
}