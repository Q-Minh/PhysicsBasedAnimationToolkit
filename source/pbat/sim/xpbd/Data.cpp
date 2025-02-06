#include "Data.h"

#include "pbat/physics/HyperElasticity.h"

#include <Eigen/LU>
#include <exception>
#include <fmt/format.h>
#include <string>
#include <tbb/parallel_for.h>

namespace pbat {
namespace sim {
namespace xpbd {

Data& Data::WithVolumeMesh(
    Eigen::Ref<MatrixX const> const& Vin,
    Eigen::Ref<IndexMatrixX const> const& Ein)
{
    this->x  = Vin;
    this->xt = Vin;
    this->T  = Ein;
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

Data& Data::WithBodies(Eigen::Ref<IndexVectorX const> const& BVin)
{
    this->BV = BVin;
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

Data& Data::WithMassInverse(Eigen::Ref<VectorX const> const& minvIn)
{
    this->minv = minvIn;
    return *this;
}

Data& Data::WithElasticMaterial(Eigen::Ref<MatrixX const> const& lameIn)
{
    this->lame = lameIn;
    return *this;
}

Data& Data::WithCollisionPenalties(Eigen::Ref<VectorX const> const& muVin)
{
    this->muV = muVin;
    return *this;
}

Data& Data::WithFrictionCoefficients(Scalar muSin, Scalar muDin)
{
    this->muS = muSin;
    this->muD = muDin;
    return *this;
}

Data& Data::WithActiveSetUpdateFrequency(Index frequency)
{
    this->mActiveSetUpdateFrequency = frequency;
    return *this;
}

Data& Data::WithDamping(Eigen::Ref<VectorX> const& betaIn, EConstraint constraint)
{
    this->beta[static_cast<std::size_t>(constraint)] = betaIn;
    return *this;
}

Data& Data::WithCompliance(Eigen::Ref<VectorX> const& alphaIn, EConstraint constraint)
{
    this->alpha[static_cast<std::size_t>(constraint)] = alphaIn;
    return *this;
}

Data& Data::WithPartitions(std::vector<Index> const& PptrIn, std::vector<Index> const& PadjIn)
{
    this->Pptr = PptrIn;
    this->Padj = PadjIn;
    return *this;
}

Data& Data::WithClusterPartitions(
    std::vector<Index> const& SGptrIn,
    std::vector<Index> const& SGadjIn,
    std::vector<Index> const& CptrIn,
    std::vector<Index> const& CadjIn)
{
    this->SGptr = SGptrIn;
    this->SGadj = SGadjIn;
    this->Cptr  = CptrIn;
    this->Cadj  = CadjIn;
    return *this;
}

Data& Data::WithDirichletConstrainedVertices(IndexVectorX const& dbcIn)
{
    this->dbc = dbcIn;
    return *this;
}

Data& Data::Construct(bool bValidate)
{
    // Set particle dynamics
    if (v.size() == 0)
    {
        v.setZero(x.rows(), x.cols());
    }
    if (aext.size() == 0)
    {
        aext.setZero(x.rows(), x.cols());
        aext.bottomRows(1).setConstant(Scalar(-9.81));
    }
    if (minv.size() == 0)
    {
        minv.setConstant(x.cols(), Scalar(1e-3));
    }
    if (BV.size() == 0)
    {
        BV.setZero(x.cols());
    }
    xb = x;
    // Enforce Dirichlet boundary conditions
    minv(dbc).setZero();
    v(Eigen::placeholders::all, dbc).setZero();
    aext(Eigen::placeholders::all, dbc).setZero();
    // Set elastic material data
    if (lame.size() == 0)
    {
        lame.setZero(2, T.cols());
        auto const [lmu, llambda] = physics::LameCoefficients(Scalar(1e6), Scalar(0.45));
        lame.row(0).setConstant(lmu);
        lame.row(1).setConstant(llambda);
    }
    DmInv.resize(3, 3 * T.cols());
    auto snhConstraintId = static_cast<std::size_t>(EConstraint::StableNeoHookean);
    alpha[snhConstraintId].resize(2 * T.cols());
    gammaSNH.resize(T.cols());
    tbb::parallel_for(Index(0), T.cols(), [&](Index t) {
        // Load vertex positions of element c
        IndexVector<4> v = T.col(t);
        Matrix<3, 4> xc  = x(Eigen::placeholders::all, v);
        // Compute shape matrix and its inverse
        Matrix<3, 3> Ds = xc.block<3, 3>(0, 1).colwise() - xc.col(0);
        auto DmInvC     = DmInv.block<3, 3>(0, t * 3);
        DmInvC          = Ds.inverse();
        // Compute constraint compliance
        Scalar const tetVolume = Ds.determinant() / Scalar(6);
        auto alphat            = alpha[snhConstraintId].segment<2>(2 * t);
        auto lamet             = lame.col(t).segment<2>(0);
        alphat                 = Scalar(1) / (lamet * tetVolume).array();
        // Compute rest stability
        gammaSNH(t) = Scalar(1) + lamet(0) / lamet(1);
    });
    if (beta[snhConstraintId].size() == 0)
    {
        beta[snhConstraintId].setZero(2 * T.cols());
    }
    lambda[snhConstraintId].setZero(2 * T.cols());
    // Set contact data
    auto collisionConstraintId = static_cast<std::size_t>(EConstraint::Collision);
    if (alpha[collisionConstraintId].size() == 0)
    {
        alpha[collisionConstraintId].setConstant(V.size(), Scalar(0));
    }
    if (beta[collisionConstraintId].size() == 0)
    {
        beta[collisionConstraintId].setConstant(V.size(), Scalar(0));
    }
    lambda[collisionConstraintId].setZero(V.size());
    if (muV.size() == 0)
    {
        muV.setOnes(V.size());
    }

    // Throw error if ill-formed Data
    if (bValidate)
    {
        // clang-format off
        bool const bPerParticleQuantityDimensionsValid = 
            x.cols() == xt.cols() and
            x.cols() == v.cols() and
            x.cols() == aext.cols() and
            x.cols() == minv.size() and 
            x.rows() == xt.rows() and
            x.rows() == v.rows() and
            x.rows() == aext.rows() and
            x.rows() == 3;
        // clang-format on
        if (not bPerParticleQuantityDimensionsValid)
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
            lame.rows() == 2 and 
            lame.cols() == T.cols();
        // clang-format on
        if (not bElementDimensionsValid)
        {
            std::string const what = fmt::format(
                "With #elements={0}, expected T=4x{0}, lame=2x{0}",
                T.cols(),
                T.cols() * 3);
            throw std::invalid_argument(what);
        }
        bool const bMultibodyContactSystemValid = BV.size() == x.cols() and muV.size() == V.size();
        if (not bMultibodyContactSystemValid)
        {
            std::string const what =
                fmt::format("Expected BV.size()={0}, muV.size()={1}", x.cols(), V.size());
            throw std::invalid_argument(what);
        }
    }
    return *this;
}

} // namespace xpbd
} // namespace sim
} // namespace pbat