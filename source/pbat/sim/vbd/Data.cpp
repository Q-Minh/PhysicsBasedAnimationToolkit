#include "Data.h"

#include "pbat/physics/HyperElasticity.h"

#include <algorithm>
#include <exception>
#include <fmt/format.h>
#include <string>

namespace pbat {
namespace sim {
namespace vbd {

Data& Data::WithVolumeMesh(
    Eigen::Ref<MatrixX const> const& Vin,
    Eigen::Ref<IndexMatrixX const> const& Ein)
{
    this->x = Vin;
    this->T = Ein;
    return *this;
}

Data& Data::WithSurfaceMesh(
    Eigen::Ref<IndexVectorX const> const& Vin,
    Eigen::Ref<IndexMatrixX const> const& Fin)
{
    this->V = Vin;
    this->F = Fin;
    return *this;
}

Data& Data::WithVelocity(Eigen::Ref<MatrixX const> const& vIn)
{
    this->v = vIn;
    return *this;
}

Data& Data::WithAcceleration(Eigen::Ref<MatrixX const> const& aextIn)
{
    this->aext = aextIn;
    return *this;
}

Data& Data::WithMass(Eigen::Ref<VectorX const> const& mIn)
{
    this->m = mIn;
    return *this;
}

Data& Data::WithQuadrature(
    Eigen::Ref<VectorX const> const& wgIn,
    Eigen::Ref<MatrixX const> const& GPIn,
    Eigen::Ref<MatrixX const> const& lameIn)
{
    this->wg   = wgIn;
    this->GP   = GPIn;
    this->lame = lameIn;
    return *this;
}

Data& Data::WithVertexAdjacency(
    Eigen::Ref<IndexVectorX const> const& GVGpIn,
    Eigen::Ref<IndexVectorX const> const& GVGgIn,
    Eigen::Ref<IndexVectorX const> const& GVGeIn,
    Eigen::Ref<IndexVectorX const> const& GVGilocalIn)
{
    this->GVGp      = GVGpIn;
    this->GVGg      = GVGgIn;
    this->GVGe      = GVGeIn;
    this->GVGilocal = GVGilocalIn;
    return *this;
}

Data& Data::WithPartitions(
    Eigen::Ref<IndexVectorX const> const& PptrIn,
    Eigen::Ref<IndexVectorX const> const& PadjIn)
{
    this->Pptr = PptrIn;
    this->Padj = PadjIn;
    return *this;
}

Data& Data::WithDirichletConstrainedVertices(IndexVectorX const& dbcIn, bool bDbcSorted)
{
    this->dbc = dbcIn;
    if (not bDbcSorted)
    {
        std::sort(this->dbc.begin(), this->dbc.end());
    }
    return *this;
}

Data& Data::WithInitializationStrategy(EInitializationStrategy strategyIn)
{
    this->strategy = strategyIn;
    return *this;
}

Data& Data::WithRayleighDamping(Scalar kDIn)
{
    this->kD = kDIn;
    return *this;
}

Data& Data::WithCollisionPenalty(Scalar kCIn)
{
    this->kC = kCIn;
    return *this;
}

Data& Data::WithHessianDeterminantZeroUnder(Scalar zero)
{
    this->detHZero = zero;
    return *this;
}

Data& Data::Construct(bool bValidate)
{
    if (xt.size() == 0)
    {
        xt = x;
    }
    if (v.size() == 0)
    {
        v.setZero(x.rows(), x.cols());
    }
    if (m.size() == 0)
    {
        m.setConstant(x.cols(), Scalar(1e3));
    }
    if (aext.size() == 0)
    {
        aext.resizeLike(x);
        aext.colwise() = Vector<3>{Scalar(0), Scalar(0), Scalar(-9.81)};
    }
    xtilde.resizeLike(x);
    xchebm2.resizeLike(x);
    xchebm1.resizeLike(x);
    vt.resizeLike(x);
    if (lame.size() == 0)
    {
        auto const [mu, lambda] = physics::LameCoefficients(Scalar(1e6), Scalar(0.45));
        lame.resize(2, T.cols());
        lame.row(0).setConstant(mu);
        lame.row(1).setConstant(lambda);
    }
    // Constrained vertices must not move
    v(Eigen::placeholders::all, dbc).setZero();
    aext(Eigen::placeholders::all, dbc).setZero();

    if (bValidate)
    {
        // clang-format off
        bool const bPerVertexQuantityDimensionsValid = 
            x.cols() == xt.cols() and
            x.cols() == v.cols() and
            x.cols() == aext.cols() and
            x.cols() == m.size() and 
            x.rows() == xt.rows() and
            x.rows() == v.rows() and
            x.rows() == aext.rows() and
            x.rows() == 3;
        // clang-format on
        if (not bPerVertexQuantityDimensionsValid)
        {
            std::string const what = fmt::format(
                "x, v, aext and m must have same #columns={} as x, and "
                "3 rows (except m)",
                x.cols());
            throw std::invalid_argument(what);
        }
        // clang-format off
        bool const bElementDimensionsValid = 
            T.rows()    == 4 and 
            T.cols()    == wg.size() and 
            GP.rows()   == 4 and 
            GP.cols()   == T.cols()*3 and 
            lame.rows() == 2 and 
            lame.cols() == T.cols();
        // clang-format on
        if (not bElementDimensionsValid)
        {
            std::string const what = fmt::format(
                "With #elements={0}, expected T=4x{0}, wg=1x{0}, GP=4x{1}, lame=2x{0}",
                T.cols(),
                T.cols() * 3);
            throw std::invalid_argument(what);
        }
        // clang-format off
        bool const bAdjacencyStructuresValid = 
            GVGp.size() == (x.cols() + 1) and 
            GVGg.size() == GVGp(Eigen::placeholders::last) and 
            GVGg.size() == GVGe.size() and 
            GVGg.size() == GVGilocal.size();
        // clang-format on
        if (not bAdjacencyStructuresValid)
        {
            std::string const what = fmt::format(
                "Expected vertex-element adjacency with prefix GVGp of size={}, "
                "and same sizes for GVGg, GVGe and GVGilocal",
                x.cols() + 1);
            throw std::invalid_argument(what);
        }
    }
    return *this;
}

} // namespace vbd
} // namespace sim
} // namespace pbat