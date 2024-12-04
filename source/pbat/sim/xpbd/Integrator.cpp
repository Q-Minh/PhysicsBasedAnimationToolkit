#include "Integrator.h"

#include "Kernels.h"
#include "pbat/common/Eigen.h"
#include "pbat/math/linalg/mini/Mini.h"
#include "pbat/profiling/Profiling.h"

#include <tbb/parallel_for.h>
#include <type_traits>

namespace pbat {
namespace sim {
namespace xpbd {

Integrator::Integrator(Data dataIn)
    : data(std::move(dataIn)),
      mTetrahedralBvh(data.x, data.T),
      mTriangleBvh(data.x, data.F),
      mParticlesInContact()
{
    auto const nCollisionVertices = static_cast<std::size_t>(data.V.size());
    mParticlesInContact.reserve(nCollisionVertices);
}

void Integrator::Step(Scalar dt, Index iterations, Index substeps)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.xpbd.Integrator.Step");

    Scalar sdt            = dt / (static_cast<Scalar>(substeps));
    Scalar sdt2           = sdt * sdt;
    auto const nParticles = data.x.cols();
    using IndexType       = std::remove_const_t<decltype(nParticles)>;
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::ToEigen;

    // Discrete collision detection
    bool constexpr bParallelize{true};
    mTetrahedralBvh.Update();
    mTriangleBvh.Update();
    // Find penetrating particles
    auto const fCullPointTet = [&](Index i, IndexVector<4> const& tet) {
        // clang-format off
        return data.V(i) == tet(0) or 
               data.V(i) == tet(1) or 
               data.V(i) == tet(2) or
               data.V(i) == tet(3);
        // clang-format on
    };
    IndexVectorX tetsInContact = mTetrahedralBvh.PrimitivesContainingPoints(
        data.x(Eigen::placeholders::all, data.V),
        fCullPointTet,
        bParallelize);
    mParticlesInContact.clear();
    for (auto i = 0; i < tetsInContact.size(); ++i)
    {
        if (tetsInContact(i) >= Index(0))
        {
            mParticlesInContact.push_back(static_cast<Index>(i));
        }
    }
    Index const nParticlesInContact = static_cast<Index>(mParticlesInContact.size());
    // Find nearest boundary face
    auto const fCullPointTriangle = [&](Index i, IndexVector<3> const& tri) {
        auto iStl = static_cast<std::size_t>(i);
        return data.BV(data.V(mParticlesInContact[iStl])) == data.BV(tri(0));
    };
    auto const [mTrianglesInContact, mSquaredDistancesToTriangles] =
        mTriangleBvh.NearestPrimitivesToPoints(
            data.x(Eigen::placeholders::all, data.V(mParticlesInContact)),
            fCullPointTriangle,
            bParallelize);
    // Dynamics integration
    for (auto s = 0; s < substeps; ++s)
    {
        // Store previous positions
        data.xt = data.x;
        // Reset lagrange multipliers
        for (auto& lambda : data.lambda)
            lambda.setZero();
        // Initialize constraint solve
        tbb::parallel_for(IndexType(0), nParticles, [&](IndexType i) {
            auto x = kernels::InitialPosition(
                FromEigen(data.xt.col(i).head<3>()),
                FromEigen(data.v.col(i).head<3>()),
                FromEigen(data.aext.col(i).head<3>()),
                sdt,
                sdt2);
            data.x.col(i) = ToEigen(x);
        });
        // Constraint loop
        for (auto k = 0; k < iterations; ++k)
        {
            // Solve tetrahedral (elasticity) constraints
            bool const bHasClusteredConstraintPartitions = not data.SGptr.empty();
            if (bHasClusteredConstraintPartitions)
            {
                ProjectClusteredBlockNeoHookeanConstraints(sdt, sdt2);
            }
            else
            {
                ProjectBlockNeoHookeanConstraints(sdt, sdt2);
            }

            // Solve contact constraints
            auto& alphaContact  = data.alpha[static_cast<int>(EConstraint::Collision)];
            auto& betaContact   = data.beta[static_cast<int>(EConstraint::Collision)];
            auto& lambdaContact = data.lambda[static_cast<int>(EConstraint::Collision)];
            tbb::parallel_for(Index(0), nParticlesInContact, [&, dt = sdt, dt2 = sdt2](Index c) {
                auto cStl                    = static_cast<std::size_t>(c);
                auto sv                      = mParticlesInContact[cStl];
                auto v                       = data.V(mParticlesInContact[cStl]);
                auto f                       = mTrianglesInContact(c);
                IndexVector<3> fv            = data.F.col(f);
                Scalar minvv                 = data.minv(v);
                mini::SVector<Scalar, 3> xvt = FromEigen(data.xt.col(v).head<3>());
                mini::SMatrix<Scalar, 3, 3> xft =
                    FromEigen(data.xt(Eigen::placeholders::all, fv).block<3, 3>(0, 0));
                mini::SMatrix<Scalar, 3, 3> xf =
                    FromEigen(data.x(Eigen::placeholders::all, fv).block<3, 3>(0, 0));
                mini::SVector<Scalar, 3> xv = FromEigen(data.x.col(v).head<3>());
                Scalar atildec              = alphaContact(c) / dt2;
                Scalar gammac               = atildec * betaContact(c) * dt;
                Scalar lambdac              = lambdaContact(c);
                Scalar muc                  = data.muV(sv);

                bool const bProject = kernels::ProjectVertexTriangle(
                    minvv,
                    xvt,
                    xft,
                    xf,
                    muc,
                    data.muS,
                    data.muD,
                    atildec,
                    gammac,
                    lambdac,
                    xv);
                if (bProject)
                {
                    lambdaContact(c) = lambdac;
                }
                data.xb.col(v) = ToEigen(xv);
            });
            data.x(Eigen::placeholders::all, data.V(mParticlesInContact)) =
                data.xb(Eigen::placeholders::all, data.V(mParticlesInContact));
        }
        // Update velocities
        tbb::parallel_for(IndexType(0), nParticles, [&](IndexType i) {
            auto v = kernels::IntegrateVelocity(
                FromEigen(data.xt.col(i).head<3>()),
                FromEigen(data.x.col(i).head<3>()),
                sdt);
            data.v.col(i) = ToEigen(v);
        });
    }
}

void Integrator::ProjectBlockNeoHookeanConstraints(Scalar dt, Scalar dt2)
{
    auto const nPartitions = static_cast<Index>(data.Pptr.size()) - 1;
    for (auto p = 0; p < nPartitions; ++p)
    {
        auto const pStl                  = static_cast<std::size_t>(p);
        auto const pbegin                = data.Pptr[pStl];
        auto const pend                  = data.Pptr[pStl + 1];
        auto const nPartitionConstraints = static_cast<Index>(pend - pbegin);
        tbb::parallel_for(Index(0), nPartitionConstraints, [&](Index k) {
            auto c = data.Padj[static_cast<std::size_t>(pbegin) + k];
            ProjectBlockNeoHookeanConstraint(c, dt, dt2);
        });
    }
}

void Integrator::ProjectClusteredBlockNeoHookeanConstraints(Scalar dt, Scalar dt2)
{
    auto const nClusterPartitions = static_cast<Index>(data.SGptr.size()) - 1;
    for (auto cp = 0; cp < nClusterPartitions; ++cp)
    {
        auto const cpStl                = static_cast<std::size_t>(cp);
        auto const cpbegin              = data.SGptr[cpStl];
        auto const cpend                = data.SGptr[cpStl + 1];
        auto const nClustersInPartition = static_cast<Index>(cpend - cpbegin);
        tbb::parallel_for(Index(0), nClustersInPartition, [&](Index k) {
            auto cc           = data.SGadj[static_cast<std::size_t>(cpbegin) + k];
            auto const ccStl  = static_cast<std::size_t>(cc);
            auto const cbegin = data.Cptr[ccStl];
            auto const cend   = data.Cptr[ccStl + 1];
            for (auto j = cbegin; j < cend; ++j)
            {
                auto jStl = static_cast<std::size_t>(j);
                auto c    = data.Cadj[jStl];
                ProjectBlockNeoHookeanConstraint(c, dt, dt2);
            }
        });
    }
}

void Integrator::ProjectBlockNeoHookeanConstraint(Index c, Scalar dt, Scalar dt2)
{
    using namespace math::linalg;
    using mini::FromEigen;
    using mini::ToEigen;
    auto const& alphaSNH = data.alpha[static_cast<int>(EConstraint::StableNeoHookean)];
    auto const& betaSNH  = data.beta[static_cast<int>(EConstraint::StableNeoHookean)];
    auto& lambdaSNH      = data.lambda[static_cast<int>(EConstraint::StableNeoHookean)];
    // Gather constraint data
    auto vinds                       = data.T.col(c);
    mini::SVector<Scalar, 4> minvc   = FromEigen(data.minv(vinds).head<4>());
    mini::SVector<Scalar, 2> atildec = FromEigen(alphaSNH.segment<2>(2 * c)) / dt2;
    mini::SVector<Scalar, 2> betac   = FromEigen(betaSNH.segment<2>(2 * c));
    mini::SVector<Scalar, 2> gammac{atildec(0) * betac(0) * dt, atildec(1) * betac(1) * dt};
    Scalar gammaSNHc                   = data.gammaSNH(c);
    mini::SMatrix<Scalar, 3, 3> DmInvc = FromEigen(data.DmInv.block<3, 3>(0, 3 * c));
    mini::SMatrix<Scalar, 3, 4> xtc =
        FromEigen(data.xt(Eigen::placeholders::all, vinds).block<3, 4>(0, 0));
    mini::SMatrix<Scalar, 3, 4> xc =
        FromEigen(data.x(Eigen::placeholders::all, vinds).block<3, 4>(0, 0));
    mini::SVector<Scalar, 2> lambdac = FromEigen(lambdaSNH.segment<2>(2 * c));
    // Project constraints
    kernels::ProjectBlockNeoHookean(minvc, DmInvc, gammaSNHc, atildec, gammac, xtc, lambdac, xc);
    // Update solution
    lambdaSNH.segment<2>(2 * c)             = ToEigen(lambdac);
    data.x(Eigen::placeholders::all, vinds) = ToEigen(xc);
}

} // namespace xpbd
} // namespace sim
} // namespace pbat

#include <doctest/doctest.h>

TEST_CASE("[sim][xpbd] Integrator")
{
    // Arrange
    // Cube mesh
    using IndexType  = pbat::Index;
    using ScalarType = pbat::Scalar;
    pbat::MatrixX P(3, 8);
    pbat::IndexVectorX V(8);
    pbat::IndexMatrixX T(4, 5);
    pbat::IndexMatrixX F(3, 12);
    // clang-format off
    P << 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f;
    T << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    F << 0, 1, 1, 3, 3, 2, 2, 0, 0, 0, 4, 5,
         1, 5, 3, 7, 2, 6, 0, 4, 3, 2, 5, 7,
         4, 4, 5, 5, 7, 7, 6, 6, 1, 3, 6, 6;
    // clang-format on
    V.setLinSpaced(0, static_cast<IndexType>(P.cols() - 1));
    std::vector<IndexType> Pptr({0, 1, 2, 3, 4, 5});
    std::vector<IndexType> Padj({0, 1, 2, 3, 4});
    // Problem parameters
    auto constexpr dt         = ScalarType{1e-2};
    auto constexpr substeps   = 20;
    auto constexpr iterations = 1;

    // Act
    using pbat::common::ToEigen;
    using pbat::sim::xpbd::Integrator;
    Integrator xpbd{pbat::sim::xpbd::Data()
                        .WithVolumeMesh(P, T)
                        .WithSurfaceMesh(V, F)
                        .WithPartitions(Pptr, Padj)
                        .Construct()};
    xpbd.Step(dt, iterations, substeps);

    // Assert
    auto constexpr zero                  = ScalarType(1e-4);
    pbat::MatrixX dx                     = xpbd.data.x - P;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < ScalarType(0)).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}