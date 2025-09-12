#include "Integrator.h"

#include "pbat/profiling/Profiling.h"

namespace pbat::sim::algorithm::newton {

Integrator::Integrator(Integrator const& other)
    : mConfig(other.mConfig),
      mElastoDynamics(other.mElastoDynamics),
      mNewton(other.mNewton),
      mLineSearch(other.mLineSearch),
      mTriplets(other.mTriplets),
      mInverseHessian(std::make_unique<DecompositionType>()),
      mHessian(other.mHessian),
      mGrad(other.mGrad)
{
}

Integrator& Integrator::operator=(Integrator const& other)
{
    if (this != &other)
    {
        mConfig         = other.mConfig;
        mElastoDynamics = other.mElastoDynamics;
        mNewton         = other.mNewton;
        mLineSearch     = other.mLineSearch;
        mTriplets       = other.mTriplets;
        mInverseHessian = std::make_unique<DecompositionType>();
        mHessian        = other.mHessian;
        mGrad           = other.mGrad;
    }
    return *this;
}

Integrator::Integrator(Config config, MeshSystemType meshSystem, ElastoDynamicsType elastoDynamics)
    : mConfig(std::move(config)),
      mMeshes(std::move(meshSystem)),
      mElastoDynamics(std::move(elastoDynamics)),
      mNewton(mConfig.nMaxIterations, mConfig.gtol, mElastoDynamics.x.size()),
      mLineSearch(mConfig.nMaxLineSearchIterations, mConfig.tauArmijo, mConfig.cArmijo),
      mTriplets(),
      mInverseHessian(std::make_unique<DecompositionType>()),
      mHessian(),
      mGrad(mElastoDynamics.x.size())
{
}

void Integrator::Step([[maybe_unused]] std::optional<io::Archive> archive)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step");
    ScalarType const dt  = mElastoDynamics.bdf.TimeStep();
    ScalarType const sdt = dt / mConfig.nSubsteps;
    mElastoDynamics.bdf.SetTimeStep(sdt);
    ScalarType const bt  = mElastoDynamics.bdf.BetaTilde();
    ScalarType const bt2 = bt * bt;
    // Setup Newton optimization
    auto const fPrepareDerivatives = [this]<class TDerivedX>(
                                         [[maybe_unused]] Eigen::MatrixBase<TDerivedX> const& xk) {
        PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.fPrepareDerivatives");
        mElastoDynamics.ComputeElasticEnergy(
            fem::EElementElasticityComputationFlags::Gradient |
                fem::EElementElasticityComputationFlags::Hessian,
            fem::EHyperElasticSpdCorrection::Absolute);
        // TODO: Compute constraint derivatives
    };
    auto const fObjective =
        [&]<class TDerivedX>([[maybe_unused]] Eigen::MatrixBase<TDerivedX> const& xk) {
            PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.fObjective");
            auto M      = mElastoDynamics.M();
            auto xtilde = mElastoDynamics.xtilde.reshaped();
            mElastoDynamics.ComputeElasticEnergy(
                fem::EElementElasticityComputationFlags::Potential,
                fem::EHyperElasticSpdCorrection::None);
            ScalarType const U = fem::HyperElasticPotential(mElastoDynamics.UgU);
            ScalarType const K = ScalarType(0.5) * (xk - xtilde).cwiseSquare().dot(M);
            return K + bt2 * U /*+ constraint potential*/;
        };
    auto const fGradient =
        [&]<class TDerivedX>([[maybe_unused]] Eigen::MatrixBase<TDerivedX> const& xk) {
            PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.fGradient");
            auto M      = mElastoDynamics.M();
            auto xtilde = mElastoDynamics.xtilde.reshaped();
            mGrad.setZero();
            fem::ToHyperElasticGradient(
                mElastoDynamics.mesh,
                mElastoDynamics.egU,
                mElastoDynamics.GgU,
                mGrad);
            mGrad *= bt2;
            mGrad += M.asDiagonal() * (xk - xtilde);
            // mGrad += constraint gradient;
            auto nNodes = mElastoDynamics.mesh.X.cols();
            auto dinds  = mElastoDynamics.DirichletNodes();
            auto all    = Eigen::placeholders::all;
            mGrad.reshaped(kDims, nNodes)(all, dinds).setZero();
            return mGrad;
        };
    auto const fHessInvProd = [&]<class TDerivedX, class TDerivedG>(
                                  [[maybe_unused]] Eigen::MatrixBase<TDerivedX> const& xk,
                                  Eigen::MatrixBase<TDerivedG> const& gk) {
        PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.fHessInvProd");
        AssembleHessian(bt2);
        mInverseHessian->compute(mHessian);
        return mInverseHessian->solve(gk);
    };
    // Time integration optimization with substepping
    // TODO: Add Augmented Lagrangian loop for constraints
    // ...
    for (auto s = 0; s < mConfig.nSubsteps; ++s)
    {
        PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.Step.Substep");
        mElastoDynamics.bdf.ConstructEquations();
        mElastoDynamics.SetupTimeIntegrationOptimization();
        auto x    = mElastoDynamics.x.reshaped();
        auto v    = mElastoDynamics.v.reshaped();
        auto xbdf = mElastoDynamics.bdf.Inertia(0);
        mNewton.Solve(fPrepareDerivatives, fObjective, fGradient, fHessInvProd, x, mLineSearch);
        v = (x + xbdf) / bt;
        mElastoDynamics.bdf.Step(x, v);
    }
    mElastoDynamics.bdf.SetTimeStep(dt);
}

void Integrator::AssembleHessian(ScalarType bt2)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.newton.Integrator.AssembleHessian");
    auto const M                   = mElastoDynamics.M();
    auto const nDofs               = M.size();
    auto const nMassTriplets       = M.size();
    auto const nElasticityTriplets = mElastoDynamics.HgU.size();
    // auto nConstraintTriplets = ;
    mTriplets.clear();
    mTriplets.reserve(
        static_cast<std::size_t>(nMassTriplets + nElasticityTriplets /*+ nConstraintTriplets*/));
    // Mass matrix
    for (auto i = 0; i < nDofs; ++i)
        mTriplets.emplace_back(i, i, M(i));
    // Stiffness matrix
    auto const nElasticityQuadPts = mElastoDynamics.wgU.size();
    auto constexpr kElemNodes     = ElastoDynamicsType::ElementType::kNodes;
    for (auto g = 0; g < nElasticityQuadPts; ++g)
    {
        auto const HgUg =
            mElastoDynamics.HgU.middleCols<kElemNodes * kDims>(g * kElemNodes * kDims);
        IndexType const e = mElastoDynamics.egU(g);
        auto const nodes  = mElastoDynamics.mesh.E.col(e);
        for (auto kj = 0; kj < kElemNodes; ++kj)
        {
            auto const nj = nodes(kj);
            for (auto ki = 0; ki < kElemNodes; ++ki)
            {
                auto const ni     = nodes(ki);
                auto const HgUgij = HgUg.block<kDims, kDims>(ki * kDims, kj * kDims);
                for (auto dj = 0; dj < kDims; ++dj)
                {
                    for (auto di = 0; di < kDims; ++di)
                    {
                        auto const i    = ni * kDims + di;
                        auto const j    = nj * kDims + dj;
                        auto const HUij = bt2 * HgUgij(di, dj);
                        mTriplets.emplace_back(i, j, HUij);
                    }
                }
            }
        }
    }
    // Remove lower triangular part and Dirichlet off-diagonal entries
    std::sort(
        mTriplets.begin(),
        mTriplets.end(),
        [](Eigen::Triplet<ScalarType, IndexType> const& a,
           Eigen::Triplet<ScalarType, IndexType> const& b) {
            /** From Eigen::SparseMatrix::setFromSortedTriplets():
             * Two triplets `a` and `b` are
             * appropriately ordered if: \code ColMajor: ((a.col() != b.col()) ? (a.col() < b.col())
             * : (a.row() < b.row()) RowMajor: ((a.row() != b.row()) ? (a.row() < b.row()) :
             * (a.col() < b.col()) \endcode
             */
            bool const bColLess = a.col() < b.col();
            bool const bRowLess = a.row() < b.row();
            bool const bSameCol = a.col() == b.col();
            return (not bSameCol and bColLess) or (bSameCol and bRowLess);
        });
    auto removeIt = std::remove_if(
        mTriplets.begin(),
        mTriplets.end(),
        [this](Eigen::Triplet<ScalarType, IndexType> const& Hij) {
            bool const bIsLowerTriangular = Hij.row() > Hij.col();
            bool const bIsDirichletRow    = mElastoDynamics.IsDirichletDof(Hij.row());
            bool const bIsDirichletCol    = mElastoDynamics.IsDirichletDof(Hij.col());
            bool const bIsDirichletOffDiag =
                (Hij.row() != Hij.col()) and (bIsDirichletRow or bIsDirichletCol);
            return bIsLowerTriangular or bIsDirichletOffDiag;
        });
    mTriplets.erase(removeIt, mTriplets.end());
    // Assemble
    mHessian.resize(nDofs, nDofs);
    mHessian.setFromSortedTriplets(mTriplets.begin(), mTriplets.end());
    // Set Dirichlet diagonals to identity
    for (auto nd : mElastoDynamics.DirichletNodes())
    {
        for (auto d = 0; d < kDims; ++d)
        {
            auto const id             = nd * kDims + d;
            mHessian.coeffRef(id, id) = ScalarType{1};
        }
    }
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
    Integrator integrator{
        Config{},
        Integrator::MeshSystemType{V, C},
        Integrator::ElastoDynamicsType{V, C}};
    MatrixX x0 = integrator.GetElastoDynamics().x;
    // Act
    integrator.Step();
    // Assert
    auto constexpr zero                  = Scalar{1e-4};
    MatrixX dx                           = integrator.GetElastoDynamics().x - x0;
    bool const bVerticesFallUnderGravity = (dx.row(2).array() < Scalar{0}).all();
    CHECK(bVerticesFallUnderGravity);
    bool const bVerticesOnlyFall = (dx.topRows(2).array().abs() < zero).all();
    CHECK(bVerticesOnlyFall);
}