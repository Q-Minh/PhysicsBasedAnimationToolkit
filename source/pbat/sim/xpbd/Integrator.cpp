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
      mParticlesInContact(),
      mTrianglesInContact(),
      mTetsInContact(),
      mSquaredDistancesToTriangles()
{
    mParticlesInContact.reserve(data.V.size());
    mTrianglesInContact.reserve(data.V.size());
    mSquaredDistancesToTriangles.reserve(data.V.size());
    mTetsInContact.reserve(data.V.size());
}

void Integrator::Step(Scalar dt, Index iterations, Index substeps)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.xpbd.Integrator.Step");

    Scalar sdt            = dt / (static_cast<Scalar>(substeps));
    Scalar sdt2           = sdt * sdt;
    auto const nParticles = data.x.cols();
    auto const nTets      = data.T.cols();
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
    mTetsInContact = mTetrahedralBvh.PrimitivesContainingPoints(
        data.x(Eigen::placeholders::all, data.V),
        fCullPointTet,
        bParallelize);
    mParticlesInContact.clear();
    for (auto i = 0; i < mTetsInContact.size(); ++i)
    {
        if (mTetsInContact[i] >= Index(0))
        {
            mParticlesInContact.push_back(i);
        }
    }
    Index const nParticlesInContact = static_cast<Index>(mParticlesInContact.size());
    // Find nearest boundary face
    auto const fCullPointTriangle = [&](Index i, IndexVector<3> const& tri) {
        return data.BV(data.V(mParticlesInContact[i])) == data.BV(tri(0));
    };
    std::tie(mTrianglesInContact, mSquaredDistancesToTriangles) =
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
            auto& alphaSNH  = data.alpha[static_cast<int>(EConstraint::StableNeoHookean)];
            auto& lambdaSNH = data.lambda[static_cast<int>(EConstraint::StableNeoHookean)];
            for (auto const& partition : data.partitions)
            {
                auto const nPartitionConstraints = static_cast<IndexType>(partition.size());
                tbb::parallel_for(
                    IndexType(0),
                    nPartitionConstraints,
                    [&, dt2 = sdt2](IndexType pc) {
                        // Gather constraint data
                        auto c                         = partition[pc];
                        auto vinds                     = data.T.col(c);
                        mini::SVector<Scalar, 4> minvc = FromEigen(data.minv(vinds).head<4>());
                        mini::SVector<Scalar, 2> atildec =
                            FromEigen(alphaSNH.segment<2>(2 * c)) / dt2;
                        Scalar gammac = data.gammaSNH(c);
                        mini::SMatrix<Scalar, 3, 3> DmInv =
                            FromEigen(data.DmInv.block<3, 3>(0, 3 * c));
                        mini::SMatrix<Scalar, 3, 4> xc =
                            FromEigen(data.x(Eigen::placeholders::all, vinds).block<3, 4>(0, 0));
                        Vector<2> lambdac = lambdaSNH.segment<2>(2 * c);
                        // Project constraints
                        kernels::ProjectDeviatoric(c, minvc, atildec(0), DmInv, lambdac(0), xc);
                        kernels::ProjectHydrostatic(
                            c,
                            minvc,
                            atildec(1),
                            gammac,
                            DmInv,
                            lambdac(1),
                            xc);
                        // Update solution
                        lambdaSNH.segment<2>(2 * c)             = lambdac;
                        data.x(Eigen::placeholders::all, vinds) = ToEigen(xc);
                    });
            }

            // Solve contact constraints
            auto& alphaContact  = data.alpha[static_cast<int>(EConstraint::Collision)];
            auto& lambdaContact = data.lambda[static_cast<int>(EConstraint::Collision)];
            tbb::parallel_for(Index(0), nParticlesInContact, [&](Index c) {
                auto sv                        = mParticlesInContact[c];
                auto v                         = data.V(mParticlesInContact[c]);
                auto f                         = mTrianglesInContact[c];
                IndexVector<3> fv              = data.F.col(f);
                Scalar minvv                   = data.minv(v);
                mini::SVector<Scalar, 3> minvf = FromEigen(data.minv(fv).head<3>());
                mini::SVector<Scalar, 3> xvt   = FromEigen(data.xt.col(v).head<3>());
                mini::SMatrix<Scalar, 3, 3> xft =
                    FromEigen(data.xt(Eigen::placeholders::all, fv).block<3, 3>(0, 0));
                mini::SMatrix<Scalar, 3, 3> xf =
                    FromEigen(data.x(Eigen::placeholders::all, fv).block<3, 3>(0, 0));
                mini::SVector<Scalar, 3> xv = FromEigen(data.x.col(v).head<3>());
                Scalar atildec              = alphaContact(c) / sdt2;
                Scalar lambdac              = lambdaContact(c);
                Scalar muc                  = data.muV(sv);

                bool const bProject = kernels::ProjectVertexTriangle(
                    minvv,
                    minvf,
                    xvt,
                    xft,
                    xf,
                    muc,
                    data.muS,
                    data.muD,
                    atildec,
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
    std::vector<std::vector<IndexType>> partitions{};
    partitions.push_back({0});
    partitions.push_back({1});
    partitions.push_back({2});
    partitions.push_back({3});
    partitions.push_back({4});
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
                        .WithPartitions(partitions)
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