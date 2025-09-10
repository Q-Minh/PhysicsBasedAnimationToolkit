#include "Integrator.h"

#include "pbat/profiling/Profiling.h"

#ifdef PBAT_USE_SUITESPARSE
    #include <Eigen/CholmodSupport>
#else
    #include <Eigen/SparseCholesky>
#endif // PBAT_USE_SUITESPARSE

namespace pbat::sim::algorithm::newton {

Integrator::Integrator(Config config, ElastoDynamicsType elastoDynamics)
    : mConfig(std::move(config)),
      mElastoDynamics(std::move(elastoDynamics)),
      mNewton(mConfig.nMaxIterations, mConfig.gtol, mElastoDynamics.x.size()),
      mLineSearch(mConfig.nMaxLineSearchIterations, mConfig.tauArmijo, mConfig.cArmijo),
      mHessian(),
      mGradU(mElastoDynamics.x.size()),
      mPreconditioner()
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator");
    // Allocate about 10% of the nodes for contact
    auto const nNodes = mElastoDynamics.mesh.X.cols();
    mHessian.AllocateContacts(static_cast<std::size_t>(nNodes / 10));
}

void Integrator::Step(std::optional<io::Archive> archive)
{
#ifdef PBAT_USE_SUITESPARSE
    using DecompositionType = Eigen::CholmodDecomposition<CSCMatrix, Eigen::Lower | Eigen::Upper>;
#else
    using DecompositionType = Eigen::SimplicialLDLT<CSCMatrix, Eigen::Lower | Eigen::Upper>;
#endif
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step");
    Scalar const dt  = mElastoDynamics.bdf.TimeStep();
    Scalar const sdt = dt / mConfig.nSubsteps;
    mElastoDynamics.bdf.SetTimeStep(sdt);
    Scalar const bt  = mElastoDynamics.bdf.BetaTilde();
    Scalar const bt2 = bt * bt;
    auto const dofs =
        ((mElastoDynamics.FreeNodes().replicate(1, kDims).transpose() * kDims).colwise() +
         IndexVector<kDims>::LinSpaced(kDims, 0, kDims - 1))
            .reshaped();
    auto const M       = mElastoDynamics.M();
    auto const Mff     = M(dofs).asDiagonal();
    auto const xftilde = mElastoDynamics.xtilde.reshaped()(dofs);
    auto xk            = mElastoDynamics.x.reshaped();
    auto vk            = mElastoDynamics.v.reshaped();
    auto xf            = xk(dofs);
    auto vf            = vk(dofs);
    // f, grad(f), hess(f)
    auto const fObjective = [&]<class TDerivedX>(Eigen::MatrixBase<TDerivedX> const& xfk) {
        PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.fObjective");
        auto dfx = xfk - xftilde;
        fem::ToElementElasticity<ElastoDynamicsType::ElasticEnergyType>(
            mElastoDynamics.mesh,
            mElastoDynamics.egU,
            mElastoDynamics.wgU,
            mElastoDynamics.GNegU,
            mElastoDynamics.lamegU.row(0),
            mElastoDynamics.lamegU.row(1),
            mElastoDynamics.x.reshaped(),
            mElastoDynamics.UgU,
            mElastoDynamics.GgU,
            mElastoDynamics.HgU,
            fem::EElementElasticityComputationFlags::Potential,
            fem::EHyperElasticSpdCorrection::None);
        Scalar const U = fem::HyperElasticPotential(mElastoDynamics.UgU);
        return Scalar(0.5) * dfx.dot(Mff * dfx) + bt2 * U;
    };
    auto const fGradient = [&]<class TDerivedX>(Eigen::MatrixBase<TDerivedX> const& xfk) {
        PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.fGradient");
        fem::ToElementElasticity<ElastoDynamicsType::ElasticEnergyType>(
            mElastoDynamics.mesh,
            mElastoDynamics.egU,
            mElastoDynamics.wgU,
            mElastoDynamics.GNegU,
            mElastoDynamics.lamegU.row(0),
            mElastoDynamics.lamegU.row(1),
            xfk.derived(),
            mElastoDynamics.UgU,
            mElastoDynamics.GgU,
            mElastoDynamics.HgU,
            fem::EElementElasticityComputationFlags::Gradient,
            fem::EHyperElasticSpdCorrection::None);
        fem::ToHyperElasticGradient(
            mElastoDynamics.mesh,
            mElastoDynamics.egU,
            mElastoDynamics.GgU,
            mGradU);
        return Mff * (xfk - xftilde) + bt2 * mGradU(dofs) /*+ contact gradient*/;
    };
    DecompositionType HffInv{};
    auto const fHessInvProd = [&]<class TDerivedX, class TDerivedG>(
                                  [[maybe_unused]] Eigen::MatrixBase<TDerivedX> const& xfk,
                                  Eigen::MatrixBase<TDerivedG> const& gfk) {
        PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.fHessInvProd");
        fem::ToElementElasticity<ElastoDynamicsType::ElasticEnergyType>(
            mElastoDynamics.mesh,
            mElastoDynamics.egU,
            mElastoDynamics.wgU,
            mElastoDynamics.GNegU,
            mElastoDynamics.lamegU.row(0),
            mElastoDynamics.lamegU.row(1),
            xfk.derived(),
            mElastoDynamics.UgU,
            mElastoDynamics.GgU,
            mElastoDynamics.HgU,
            fem::EElementElasticityComputationFlags::Hessian,
            fem::EHyperElasticSpdCorrection::None);
        // mHessian.SetSparsityPattern(U.GH.Pattern());
        // mHessian.ConstructContactLessHessian(M, bt2, U);
        mHessian.ConstructContactHessian();
        mHessian.ImposeDirichletConstraints(dofs);
        // Fold contact hessian into the contact-less hessian
        mHessian.HNC += mHessian.HC;
        HffInv.compute(mHessian.HNC);
        return HffInv.solve(gfk);
    };
    // Time integration optimization with substepping
    for (auto s = 0; s < mConfig.nSubsteps; ++s)
    {
        PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.Substep");
        mElastoDynamics.bdf.ConstructEquations();
        mElastoDynamics.SetupTimeIntegrationOptimization();
        mNewton.Solve(fObjective, fGradient, fHessInvProd, xf, mLineSearch);
        vf = (xf + mElastoDynamics.bdf.Inertia(0)) / bt;
        mElastoDynamics.bdf.Step(xk, vk);
    }
    mElastoDynamics.bdf.SetTimeStep(dt);
}

} // namespace pbat::sim::algorithm::newton

#include <doctest/doctest.h>

TEST_CASE("[sim][algorithm][newton] Integrator")
{
    using namespace pbat;
    using namespace pbat::sim::algorithm::newton;

    // Arrange
    // Cube tetrahedral mesh
    MatrixX V(3, 8);
    IndexMatrixX C(4, 5);
    // clang-format off
    V << 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 0., 1., 1., 0., 0., 1., 1.,
         0., 0., 0., 0., 1., 1., 1., 1.;
    C << 0, 3, 5, 6, 0,
         1, 2, 4, 7, 5,
         3, 0, 6, 5, 3,
         5, 6, 0, 3, 6;
    // clang-format on
    Integrator integrator{Config{}, Integrator::ElastoDynamicsType{V, C}};
    MatrixX x0 = integrator.mElastoDynamics.x;
    // Act
    integrator.Step();
    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = integrator.mElastoDynamics.x - x0;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}