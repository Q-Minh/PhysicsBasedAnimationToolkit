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
    Eigen::Ref<MatrixX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& E)
{
    this->x  = V;
    this->xt = V;
    this->T  = E;
    return *this;
}

Data& Data::WithSurfaceMesh(
    Eigen::Ref<IndexVectorX const> const& V,
    Eigen::Ref<IndexMatrixX const> const& F)
{
    this->V = V;
    this->F = F;
    return *this;
}

Data& Data::WithBodies(Eigen::Ref<IndexVectorX const> const& BV)
{
    this->BV = BV;
    return *this;
}

Data& Data::WithVelocity(Eigen::Ref<MatrixX const> const& v)
{
    this->v = v;
    return *this;
}

Data& Data::WithAcceleration(Eigen::Ref<MatrixX const> const& aext)
{
    this->aext = aext;
    return *this;
}

Data& Data::WithMassInverse(Eigen::Ref<VectorX const> const& minv)
{
    this->minv = minv;
    return *this;
}

Data& Data::WithElasticMaterial(Eigen::Ref<MatrixX const> const& lame)
{
    this->lame = lame;
    return *this;
}

Data& Data::WithFrictionCoefficients(Scalar muS, Scalar muD)
{
    this->muS = muS;
    this->muD = muD;
    return *this;
}

Data& Data::WithCompliance(Eigen::Ref<VectorX> const& alpha, EConstraint constraint)
{
    this->alpha[static_cast<int>(constraint)] = alpha;
    return *this;
}

Data& Data::WithPartitions(std::vector<std::vector<Index>> const& partitions)
{
    this->partitions = partitions;
    return *this;
}

Data& Data::WithDirichletConstrainedVertices(IndexVectorX const& dbc)
{
    this->dbc = dbc;
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
        aext.row(aext.rows() - 1).setConstant(Scalar(-9.81));
    }
    if (minv.size() == 0)
    {
        minv.setConstant(Scalar(1));
    }
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
    alpha[static_cast<int>(EConstraint::StableNeoHookean)].resize(2 * T.cols());
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
        auto alphat = alpha[static_cast<int>(EConstraint::StableNeoHookean)].segment<2>(2 * t);
        auto lamet  = lame.col(t).segment<2>(0);
        alphat      = Scalar(1) / (lamet * tetVolume).array();
        // Compute rest stability
        gammaSNH(t) = Scalar(1) + lamet(0) / lamet(1);
    });
    lambda[static_cast<int>(EConstraint::StableNeoHookean)].setZero(2 * T.cols());
    // Set contact data
    alpha[static_cast<int>(EConstraint::Collision)].setConstant(V.cols(), alphaC);
    lambda[static_cast<int>(EConstraint::Collision)].setZero(V.cols());

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
        bool const bMultibodyContactSystemValid = BV.size() == V.size();
        if (not bMultibodyContactSystemValid)
        {
            std::string const what = fmt::format(
                "With #collision vertices={0}, #collision faces={1} expected BV.size()={0}",
                V.size(),
                F.cols());
            throw std::invalid_argument(what);
        }
    }
    return *this;
}

} // namespace xpbd
} // namespace sim
} // namespace pbat