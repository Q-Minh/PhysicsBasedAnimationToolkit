#include "Data.h"

#include "pbat/fem/Jacobian.h"
#include "pbat/fem/MassMatrix.h"
#include "pbat/fem/ShapeFunctions.h"
#include "pbat/graph/Adjacency.h"
#include "pbat/graph/Mesh.h"
#include "pbat/physics/HyperElasticity.h"

#include <algorithm>
#include <exception>
#include <fmt/format.h>
#include <string>
#include <unordered_set>

namespace pbat {
namespace sim {
namespace vbd {

Data& Data::WithVolumeMesh(
    Eigen::Ref<MatrixX const> const& Vin,
    Eigen::Ref<IndexMatrixX const> const& Ein)
{
    this->mesh = Mesh(Vin, Ein);
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
    // Vertex data
    x = mesh.X;
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
    xchebm2.resizeLike(x);
    xchebm1.resizeLike(x);
    vt.resizeLike(x);
    // Material parameters
    if (lame.size() == 0)
    {
        auto const [mu, lambda] = physics::LameCoefficients(Scalar(1e6), Scalar(0.45));
        lame.resize(2, mesh.E.cols());
        lame.row(0).setConstant(mu);
        lame.row(1).setConstant(lambda);
    }
    if (rhoe.size() == 0)
    {
        rhoe.setConstant(mesh.E.cols(), Scalar(1e3));
    }
    MatrixX detJe = fem::DeterminantOfJacobian<2>(mesh);
    MatrixX rhog  = rhoe.transpose().replicate(detJe.rows(), 1);
    fem::MassMatrix<Mesh, 2> M(mesh, detJe, rhog, 1);
    m  = M.ToLumpedMasses();
    GP = fem::ShapeFunctionGradients<1>(mesh);
    wg = fem::InnerProductWeights<1>(mesh).reshaped();
    // Adjacency structures
    IndexMatrixX ilocal             = IndexVector<4>{0, 1, 2, 3}.replicate(1, mesh.E.cols());
    auto GVT                        = graph::MeshAdjacencyMatrix(mesh.E, ilocal, mesh.X.cols());
    GVT                             = GVT.transpose();
    std::tie(GVGp, GVGe, GVGilocal) = graph::MatrixToAdjacency(GVT);
    // Parallel partitions
    auto GVV                = graph::MeshPrimalGraph(mesh.E, mesh.X.cols());
    auto [GVVp, GVVv, GVVw] = graph::MatrixToAdjacency(GVV);
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
    }
    return *this;
}

} // namespace vbd
} // namespace sim
} // namespace pbat