#include "Data.h"

#include "Mesh.h"
#include "pbat/fem/Mass.h"
#include "pbat/fem/MeshQuadrature.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Color.h"
#include "pbat/graph/Mesh.h"
#include "pbat/physics/HyperElasticity.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <exception>
#include <fmt/format.h>
#include <string>
#include <unordered_set>

namespace pbat::sim::vbd {

Data& Data::WithVolumeMesh(
    Eigen::Ref<MatrixX const> const& Vin,
    Eigen::Ref<IndexMatrixX const> const& Ein)
{
    this->X = Vin;
    this->E = Ein;
    if (this->B.size() == 0)
    {
        this->B.setOnes(X.cols());
    }
    return *this;
}

Data& Data::WithSurfaceMesh(
    Eigen::Ref<IndexVectorX const> const& Vin,
    Eigen::Ref<IndexMatrixX const> const& Fin)
{
    this->V  = Vin;
    this->F  = Fin;
    auto FAB = X(Eigen::placeholders::all, F.row(1)) - X(Eigen::placeholders::all, F.row(0));
    auto FAC = X(Eigen::placeholders::all, F.row(2)) - X(Eigen::placeholders::all, F.row(0));
    XVA.setZero(X.cols());
    FA.resize(F.cols());
    for (auto f = 0; f < F.cols(); ++f)
    {
        auto AB      = FAB.col(f).head<3>().eval();
        auto AC      = FAC.col(f).head<3>().eval();
        auto dblarea = AB.cross(AC).norm();
        for (auto i : F.col(f))
            XVA(i) += dblarea / 6;
        FA(f) = dblarea / 2;
    }
    return *this;
}

Data& Data::WithBodies(Eigen::Ref<IndexVectorX const> const& Bin)
{
    this->B = Bin;
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

Data& Data::WithMaterial(
    Eigen::Ref<VectorX const> const& rhoeIn,
    Eigen::Ref<VectorX const> const& mue,
    Eigen::Ref<VectorX const> const& lambdae)
{
    this->rhoe = rhoeIn;
    this->lame.resize(2, mue.size());
    this->lame.row(0) = mue;
    this->lame.row(1) = lambdae;
    return *this;
}

Data& Data::WithDirichletConstrainedVertices(
    IndexVectorX const& dbcIn,
    Scalar muDin,
    bool bDbcSorted)
{
    this->dbc = dbcIn;
    this->muD = muDin;
    if (not bDbcSorted)
    {
        std::sort(this->dbc.begin(), this->dbc.end());
    }
    return *this;
}

Data& Data::WithVertexColoringStrategy(
    graph::EGreedyColorOrderingStrategy eOrderingIn,
    graph::EGreedyColorSelectionStrategy eSelectionIn)
{
    eOrdering  = eOrderingIn;
    eSelection = eSelectionIn;
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

Data& Data::WithContactParameters(Scalar muCin, Scalar muFin, Scalar epsvIn)
{
    this->muC  = muCin;
    this->muF  = muFin;
    this->epsv = epsvIn;
    return *this;
}

Data& Data::WithActiveSetUpdateFrequency(Index activeSetUpdateFrequency)
{
    this->mActiveSetUpdateFrequency = activeSetUpdateFrequency;
    return *this;
}

Data& Data::WithHessianDeterminantZeroUnder(Scalar zero)
{
    this->detHZero = zero;
    return *this;
}

Data& Data::WithChebyshevAcceleration(Scalar rhoIn)
{
    this->rho           = rhoIn;
    this->eAcceleration = EAccelerationStrategy::Chebyshev;
    return *this;
}

Data& Data::WithAndersonAcceleration(Index window)
{
    this->mAndersonWindowSize = window;
    this->eAcceleration       = EAccelerationStrategy::Anderson;
    return *this;
}

Data& Data::WithAcceleratedAnderson(Index window)
{
    this->mAndersonWindowSize = window;
    this->eAcceleration       = EAccelerationStrategy::AcceleratedAnderson;
    return *this;
}

Data& Data::WithNesterovAcceleration(Scalar L, Index start)
{
    this->mNesterovLipschitzConstant = L;
    this->mNesterovAccelerationStart = start;
    this->eAcceleration              = EAccelerationStrategy::Nesterov;
    return *this;
}

Data& Data::WithTrustRegionAcceleration(Scalar etaIn, Scalar tauIn, bool bCurvedIn)
{
    this->eta           = etaIn;
    this->tau           = tauIn;
    this->bCurved       = bCurvedIn;
    this->eAcceleration = EAccelerationStrategy::TrustRegion;
    return *this;
}

Data& Data::Construct(bool bValidate)
{
    // Vertex data
    x = X;
    if (xt.size() == 0)
    {
        xt = x;
    }
    if (v.size() == 0)
    {
        v.setZero(x.rows(), x.cols());
    }
    if (aext.size() == 0)
    {
        aext.resizeLike(x);
        aext.colwise() = Vector<3>{Scalar(0), Scalar(0), Scalar(-9.81)};
    }
    xtilde.resizeLike(x);
    vt.resizeLike(x);
    // Element data
    if (lame.size() == 0)
    {
        auto const [mu, lambda] = physics::LameCoefficients(Scalar(1e6), Scalar(0.45));
        lame.resize(2, E.cols());
        lame.row(0).setConstant(mu);
        lame.row(1).setConstant(lambda);
    }
    if (rhoe.size() == 0)
    {
        rhoe.setConstant(E.cols(), Scalar(1e3));
    }
    VolumeMesh mesh{X, E};
    auto constexpr kQuadratureOrder = 2 * VolumeMesh::ElementType::kOrder;
    auto const wgM                  = fem::MeshQuadratureWeights<kQuadratureOrder>(mesh);
    auto const egM                  = fem::MeshQuadratureElements(mesh.E, wgM);
    auto const nQuadPtsPerElement   = wgM.rows();
    auto const rhog                 = rhoe.transpose().replicate(nQuadPtsPerElement, 1).reshaped();
    auto const Ng  = fem::ElementShapeFunctions<VolumeMesh::ElementType, kQuadratureOrder>();
    auto const Neg = Ng.replicate(1, mesh.E.cols());
    auto const Meg = fem::ElementMassMatrices<VolumeMesh::ElementType>(Neg, wgM.reshaped(), rhog);
    fem::ToLumpedMassMatrix(mesh, egM.reshaped(), Meg, 1, m);
    GP = fem::ShapeFunctionGradients<1>(mesh);
    wg = fem::MeshQuadratureWeights<1>(mesh).reshaped();
    // Adjacency structures
    IndexMatrixX ilocal             = IndexVector<4>{0, 1, 2, 3}.replicate(1, mesh.E.cols());
    auto GVT                        = graph::MeshAdjacencyMatrix(mesh.E, ilocal, mesh.X.cols());
    GVT                             = GVT.transpose();
    std::tie(GVGp, GVGe, GVGilocal) = graph::MatrixToWeightedAdjacency(GVT);
    // Parallel partitions
    auto GVV                = graph::MeshPrimalGraph(mesh.E, mesh.X.cols());
    auto [GVVp, GVVv, GVVw] = graph::MatrixToWeightedAdjacency(GVV);
    colors                  = graph::GreedyColor(GVVp, GVVv, eOrdering, eSelection);
    std::tie(Pptr, Padj)    = graph::MapToAdjacency(colors);
    // Apply Dirichlet boundary conditions.
    // This is done by removing any velocity and external accelerations (i.e. external forces) on
    // Dirichet vertices. Additionally, we omit internal forces of Dirichlet vertices by removing
    // such vertices from the minimization, i.e. the parallel vertex partitions.
    v(Eigen::placeholders::all, dbc).setZero();
    aext(Eigen::placeholders::all, dbc).setZero();
    std::unordered_set<Index> D{};
    D.reserve(dbc.size() * 3ULL);
    D.insert(dbc.begin(), dbc.end());
    graph::RemoveEdges(Pptr, Padj, [&]([[maybe_unused]] Index p, Index v) {
        return D.find(v) != D.end();
    });
    // Validate user input
    if (bValidate)
    {
        // clang-format off
        bool const bPerVertexQuantityDimensionsValid = 
            x.cols() == xt.cols() and
            x.cols() == v.cols() and
            x.cols() == aext.cols() and
            x.cols() == m.size() and 
            x.cols() == B.size() and
            x.rows() == xt.rows() and
            x.rows() == v.rows() and
            x.rows() == aext.rows() and
            x.rows() == 3;
        // clang-format on
        if (not bPerVertexQuantityDimensionsValid)
        {
            std::string const what = fmt::format(
                "x, v, aext, m and B must have same #columns={} as x, and "
                "3 rows (except m and B)",
                x.cols());
            throw std::invalid_argument(what);
        }

        switch (eAcceleration)
        {
            case EAccelerationStrategy::None: break;
            case EAccelerationStrategy::Chebyshev:
                if (rho <= 0 or rho >= 1)
                {
                    throw std::invalid_argument("Expected 0 < rho < 1");
                }
                break;
            case EAccelerationStrategy::Anderson: [[fallthrough]];
            case EAccelerationStrategy::AcceleratedAnderson:
                if (mAndersonWindowSize < 1)
                {
                    throw std::invalid_argument("Expected m > 0");
                }
                break;
            case EAccelerationStrategy::Nesterov:
                if (mNesterovLipschitzConstant <= Scalar(0))
                {
                    throw std::invalid_argument("Expected L > 0");
                }
                if (mNesterovAccelerationStart < 0)
                {
                    throw std::invalid_argument("Expected start >= 0");
                }
                break;
            case EAccelerationStrategy::TrustRegion:
                if (eta < 0)
                {
                    throw std::invalid_argument("Expected eta >= 0");
                }
                if (tau <= 1)
                {
                    throw std::invalid_argument("Expected tau > 1");
                }
                break;
            default: break;
        }
    }
    return *this;
}

} // namespace pbat::sim::vbd