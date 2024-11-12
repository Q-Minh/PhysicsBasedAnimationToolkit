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
        return data.V(i) == tet(0) or data.V(i) == tet(1) or data.V(i) == tet(2) or i == tet(3);
    };
    mTetsInContact = mTetrahedralBvh.PrimitivesContainingPoints(
        data.x(Eigen::placeholders::all, data.V),
        fCullPointTet,
        bParallelize);
    Index nParticlesInContact{0};
    for (auto v = 0; v < mTetsInContact.size(); ++v)
    {
        if (mTetsInContact[v] >= Index(0))
        {
            mParticlesInContact[nParticlesInContact++] = v;
        }
    }
    // Find nearest boundary face
    auto const fCullPointTriangle = [&](Index i, IndexVector<3> const& tri) {
        return data.BV(data.V(mParticlesInContact[i])) == data.BV(tri(0));
    };
    std::tie(mTrianglesInContact, mSquaredDistancesToTriangles) =
        mTriangleBvh.NearestPrimitivesToPoints(
            data.x(
                Eigen::placeholders::all,
                common::ToEigen(mParticlesInContact).head(nParticlesInContact)),
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
            tbb::parallel_for(Index(0), nParticlesInContact, [&, dt2 = sdt2](Index vc) {
                auto v                      = data.V(mParticlesInContact[vc]);
                auto f                      = mTrianglesInContact[vc];
                IndexVector<3> fv           = data.F.col(f);
                Scalar minvv                = data.minv(v);
                Vector<3> minvf             = data.minv(fv);
                Vector<3> xvt               = data.xt.col(v);
                Matrix<3, 3> xft            = data.xt(Eigen::placeholders::all, fv);
                Matrix<3, 3> xf             = data.x(Eigen::placeholders::all, fv);
                mini::SVector<Scalar, 3> xv = FromEigen(data.x.col(v).head<3>());
                Scalar atildec              = alphaContact(vc) / dt2;
                Scalar lambdac              = lambdaContact(vc);

                bool bProject = kernels::ProjectVertexTriangle(
                    minvv,
                    FromEigen(minvf),
                    FromEigen(xvt),
                    FromEigen(xft),
                    FromEigen(xf),
                    data.muS,
                    data.muD,
                    atildec,
                    lambdac,
                    xv);
                if (bProject)
                {
                    lambdaContact(vc) = lambdac;
                    data.x.col(v)     = ToEigen(xv);
                }
            });
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