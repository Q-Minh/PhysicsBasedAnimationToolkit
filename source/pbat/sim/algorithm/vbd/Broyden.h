/**
 * @file Broyden.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Broyden's method accelerated VBD.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_BROYDEN_H
#define PBAT_SIM_ALGORITHM_VBD_BROYDEN_H

#include "Core.h"
#include "Enums.h"
#include "pbat/common/Modulo.h"
#include "pbat/io/Archive.h"
#include "pbat/profiling/Profiling.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/QR>
#include <exception>
#include <fmt/core.h>

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Broyden accelerated VBD solver parameters
 *
 * @details See @cite anderson_iterative_1965, @cite fang_two_2009
 */
struct BroydenParams
{
    Index m{5};                 ///< Window size
    Scalar epsL2Solve{0};       ///< Numerical zero threshold for least-squares solver
    Index maxL2SolverIters{-1}; ///< Maximum iterations for (iterative) least-squares solver
    EBroydenLeastSquaresSolver eL2Solver{
        EBroydenLeastSquaresSolver::COD}; ///< Least-squares solver type
    EBroydenJacobianEstimate eJacobianEstimate{
        EBroydenJacobianEstimate::Identity}; ///< Jacobian estimate strategy
    Scalar betaF{1}; ///< Rank estimate for Fk in diagonal Cauchy-Schwarz updating
    Scalar betaB{1}; ///< Rank estimate for Bk in diagonal Cauchy-Schwarz updating

    /**
     * @brief Read/Write parameters
     */
    Index k;        ///< Current iteration
    MatrixX Fk;     ///< `|# dofs| x m` residual differences
    MatrixX Xk;     ///< `|# dofs| x m` past step differences
    VectorX xkm1;   ///< `|# dofs| x 1` previous step
    VectorX fk;     ///< `|# dofs| x 1` current residual
    VectorX fkm1;   ///< `|# dofs| x 1` past residual
    VectorX gammak; ///< `m x 1` subspace residual
    /**
     * @brief Quasi-Newton jacobian estimate
     */
    MatrixX FkRowNorm2; ///< `|# dofs| x m` Cauchy-Schwarz squared norms on rows of Fk
    MatrixX Gkm;        ///< `|# dofs| x m` diag(G_{k-m})
    VectorX Sigma;      ///< `m x 1` scaled identity coefficients window
    Scalar sqrtBetaB;   ///< Cached sqrt(betaB) for diagonal Cauchy-Schwarz updating
    Scalar Fknorm2;     ///< Cached ||F_k||_F^2 for diagonal Cauchy-Schwarz updating
    Scalar Bknorm2;     ///< Cached ||B_k||_F^2 for diagonal Cauchy-Schwarz updating
    /**
     * @brief Least-squares solver
     */
    VectorX gradL2;                                      ///< `m x 1` least-squares gradient
    VectorX FkgradL2;                                    ///< `|# dofs| x 1` Fk * gradL2
    Eigen::CompleteOrthogonalDecomposition<MatrixX> cod; ///< COD solver for least-squares problem
    Eigen::ColPivHouseholderQR<MatrixX> qr;              ///< QR solver for least-squares problem
    Eigen::LeastSquaresConjugateGradient<MatrixX>
        lscg; ///< Iterative LSQR solver for least-squares problem
    
    /**
     * @brief Serialize this to archive
     * @param archive Archive to serialize to
     */
    PBAT_API void Serialize(io::Archive& archive) const;
    /**
     * @brief Deserialize this from archive
     * @param archive Archive to deserialize from
     */
    PBAT_API void Deserialize(io::Archive const& archive);
    /**
     * @brief Allocate memory for Anderson parameters
     * @param n Number of degrees of freedom
     */
    void AllocateIfNeeded(Index n)
    {
        Fk.resize(n, m);
        Xk.resize(n, m);
        xkm1.resize(n);
        fk.resize(n);
        fkm1.resize(n);
        gammak.resize(m);
        switch (eJacobianEstimate)
        {
            case EBroydenJacobianEstimate::Identity: break;
            case EBroydenJacobianEstimate::ScaledIdentity: Sigma.resize(m); break;
            case EBroydenJacobianEstimate::QuasiCauchyRelationDiagonalUpdating: {
                Gkm.resize(n, m);
            }
            break;
            case EBroydenJacobianEstimate::UsdDiagonal: {
                Gkm.resize(n, m);
            }
            break;
            case EBroydenJacobianEstimate::DiagonalCauchySchwarz: {
                FkRowNorm2.resize(n, m);
                Gkm.resize(n, m);
            }
            break;
            default: break;
        }
        switch (eL2Solver)
        {
            case EBroydenLeastSquaresSolver::COD: break;
            case EBroydenLeastSquaresSolver::QR: break;
            case EBroydenLeastSquaresSolver::LSCG: break;
            case EBroydenLeastSquaresSolver::OneStepSteepestDescent: {
                gradL2.resize(m);
                FkgradL2.resize(n);
            }
            break;
            default: break;
        }
    }
};

/**
 * @brief Initialize Broyden accelerated VBD minimization solve
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param broyden Broyden parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    BroydenParams& broyden);

/**
 * @brief One Broyden accelerated VBD minimization step
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param broyden Broyden parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, BroydenParams& broyden);

/**
 * @brief Solve FEM elasto dynamics time integration minimization problem using Broyden-accelerated
 * VBD
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param broyden Broyden parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, BroydenParams& broyden);

/**
 * @brief Integrate FEM elasto dynamics one step using Broyden-accelerated VBD as the non-linear
 * solver
 * @tparam TElasticEnergy Hyper-elastic energy model
 * @param fem Finite element elasto dynamics problem (in/out parameter)
 * @param params Solver parameters
 * @param broyden Broyden parameters
 * @pre `TElasticEnergy::kDims == 3`
 */
template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    BroydenParams& broyden);

template <physics::CHyperElasticEnergy TElasticEnergy>
void InitializeSolve(
    FemElastoDynamics<TElasticEnergy>& fem,
    Params const& params,
    BroydenParams& broyden)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Broyden.InitializeSolve");
    broyden.AllocateIfNeeded(fem.x.size());
    InitializeSolve<TElasticEnergy>(fem, params);

    switch (broyden.eJacobianEstimate)
    {
        case EBroydenJacobianEstimate::Identity: break;
        case EBroydenJacobianEstimate::ScaledIdentity: {
            broyden.Sigma.setOnes();
        }
        break;
        case EBroydenJacobianEstimate::QuasiCauchyRelationDiagonalUpdating: {
            broyden.Gkm.setOnes();
        }
        break;
        case EBroydenJacobianEstimate::UsdDiagonal: {
            broyden.Gkm.setOnes();
        }
        break;
        case EBroydenJacobianEstimate::DiagonalCauchySchwarz: {
            broyden.FkRowNorm2.setZero();
            broyden.Gkm.setOnes();
            broyden.sqrtBetaB = std::sqrt(broyden.betaB);
            broyden.Fknorm2   = 0;
            broyden.Bknorm2   = 0; // ||I||_F^2
        }
        break;
        default: break;
    }

    broyden.xkm1 = fem.x.reshaped();
    Iterate(fem, params);
    broyden.fkm1 = broyden.xkm1 - fem.x.reshaped();
    broyden.k    = 1;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Iterate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, BroydenParams& broyden)
{
    PBAT_PROFILE_NAMED_SCOPE("pbat.sim.algorithm.vbd.Broyden.Iterate");
    auto dkl = common::Modulo(broyden.k - 1, broyden.m);
    // Update (preconditioned) history
    broyden.Xk.col(dkl) = fem.x.reshaped() - broyden.xkm1;
    broyden.xkm1        = fem.x.reshaped();
    Iterate(fem, params);
    broyden.fk          = broyden.xkm1 - fem.x.reshaped();
    broyden.Fk.col(dkl) = broyden.fk - broyden.fkm1;
    broyden.fkm1        = broyden.fk;
    // Solve least-squares problem
    auto mk = std::min(broyden.m, broyden.k);
    auto Fk = broyden.Fk.leftCols(mk);
    // NOTE: The decomposition solvers (i.e. COD, QR) should use an updating scheme here instead of
    // recomputing from scratch every time (Eigen does not seem to support it), but the updating
    // scheme needs to account for pivoting as well.
    switch (broyden.eL2Solver)
    {
        case EBroydenLeastSquaresSolver::QR: {
            broyden.qr.compute(Fk);
            if (broyden.qr.info() != Eigen::ComputationInfo::Success)
            {
                throw std::runtime_error(
                    fmt::format("QR decomposition failed at iteration {}", broyden.k));
            }
            broyden.gammak.head(mk) = broyden.qr.solve(broyden.fk);
        }
        break;
        case EBroydenLeastSquaresSolver::LSCG: {
            broyden.lscg.setMaxIterations(
                broyden.maxL2SolverIters > 0 ? broyden.maxL2SolverIters : mk);
            broyden.lscg.setTolerance(broyden.epsL2Solve);
            broyden.lscg.compute(Fk);
            broyden.gammak.head(mk) = broyden.lscg.solve(broyden.fk);
        }
        break;
        case EBroydenLeastSquaresSolver::OneStepSteepestDescent: {
            broyden.gradL2       = Fk.transpose() * broyden.fk;
            broyden.FkgradL2     = Fk * broyden.gradL2;
            Scalar gradL2norm2   = broyden.gradL2.squaredNorm();
            Scalar FkgradL2norm2 = broyden.FkgradL2.squaredNorm();
            Scalar alpha = FkgradL2norm2 > Scalar(0) ? gradL2norm2 / FkgradL2norm2 : Scalar(0);
            broyden.gammak.head(mk) = alpha * broyden.gradL2;
        }
        break;
        case EBroydenLeastSquaresSolver::COD: [[fallthrough]];
        default: {
            broyden.cod.compute(Fk);
            if (broyden.cod.info() != Eigen::ComputationInfo::Success)
            {
                throw std::runtime_error(
                    fmt::format("COD decomposition failed at iteration {}", broyden.k));
            }
            broyden.gammak.head(mk) = broyden.cod.solve(broyden.fk);
        }
        break;
    }
    // Broyden step
    bool const bHasUpdatingDiagonal =
        broyden.eJacobianEstimate == EBroydenJacobianEstimate::DiagonalCauchySchwarz or
        broyden.eJacobianEstimate == EBroydenJacobianEstimate::UsdDiagonal or
        broyden.eJacobianEstimate == EBroydenJacobianEstimate::QuasiCauchyRelationDiagonalUpdating;
    bool const bHasScaledIdentityJacobian =
        broyden.eJacobianEstimate == EBroydenJacobianEstimate::ScaledIdentity;
    // NOTE: At this point, broyden.xkm1 contains x_k, while fem.x.reshaped() contains x_k + f_k
    if (bHasUpdatingDiagonal)
    {
        // x_{k+1} = x_k - G_{k-m} VBD(f_k) - (X_k - G_{k-m} VBD(F_k)) \gamma_k
        fem.x.reshaped() = broyden.xkm1 - broyden.Gkm.col(dkl).asDiagonal() * broyden.fk;
        fem.x.reshaped() -= broyden.Xk.leftCols(mk) * broyden.gammak.head(mk);
        fem.x.reshaped() +=
            broyden.Gkm.col(dkl).asDiagonal() * (broyden.Fk.leftCols(mk) * broyden.gammak.head(mk));
    }
    else if (bHasScaledIdentityJacobian)
    {
        Scalar sigma     = broyden.Sigma(dkl);
        fem.x.reshaped() = broyden.xkm1 - sigma * broyden.fk;
        fem.x.reshaped() -= broyden.Xk.leftCols(mk) * broyden.gammak.head(mk);
        fem.x.reshaped() += sigma * (broyden.Fk.leftCols(mk) * broyden.gammak.head(mk));
    }
    else
    {
        fem.x.reshaped() -= broyden.Xk.leftCols(mk) * broyden.gammak.head(mk);
        fem.x.reshaped() += broyden.Fk.leftCols(mk) * broyden.gammak.head(mk);
    }
    // Update Jacobian (inverse) estimate
    switch (broyden.eJacobianEstimate)
    {
        case EBroydenJacobianEstimate::Identity: break;
        case EBroydenJacobianEstimate::ScaledIdentity: {
            auto sk      = broyden.Xk.col(dkl);
            auto yk      = broyden.Fk.col(dkl);
            Scalar ykTyk = yk.squaredNorm();
            Scalar skTyk = sk.dot(yk);
            broyden.Sigma(dkl) =
                ykTyk > Scalar(0) ? skTyk / ykTyk : Scalar(1) /* fall back to VBD */;
        }
        break;
        case EBroydenJacobianEstimate::QuasiCauchyRelationDiagonalUpdating: {
            auto ddkl        = common::Modulo(dkl - 1, broyden.m);
            auto Hkm1        = broyden.Gkm.col(ddkl);
            auto sk          = broyden.Xk.col(dkl);
            auto yk          = broyden.Fk.col(dkl);
            Scalar skTyk     = sk.dot(yk);
            auto Dkm1        = Hkm1.cwiseInverse();
            Scalar skTDkm1sk = sk.dot(Dkm1.asDiagonal() * sk);
            auto Ek          = sk.array().square();
            Scalar trEk2     = (Ek * Ek).sum();
            if (trEk2 > Scalar(0) and skTyk > skTDkm1sk)
            {
                broyden.Gkm.col(dkl) =
                    (Dkm1.array() + ((skTyk - skTDkm1sk) / trEk2) * Ek).cwiseInverse();
            }
        }
        break;
        case EBroydenJacobianEstimate::UsdDiagonal: {
            // Compute Delta 1 and store it in Gkm.col(dkl)
            auto yk        = broyden.Fk.col(dkl);
            auto sk        = broyden.Xk.col(dkl);
            Scalar skTyk   = sk.dot(yk);
            Scalar ykTyk   = yk.squaredNorm();
            Scalar ykTsk2  = skTyk * skTyk;
            Scalar deltaki = skTyk / ykTyk;
            Scalar ykTDkyk = deltaki * ykTyk;
            broyden.Gkm.col(dkl).array() =
                deltaki + (Scalar(1) / skTyk + ykTDkyk / ykTsk2) * sk.array().square() -
                (Scalar(2) * deltaki / skTyk) * sk.array() * yk.array();
            // Compute Hkm and store it in Gkm.col(dkl)
            auto Yk        = yk.array().square();
            Scalar trYk2   = (Yk * Yk).sum();
            Scalar ykTD1yk = yk.dot(broyden.Gkm.col(dkl).asDiagonal() * yk);
            broyden.Gkm.col(dkl).array() += Scalar(1) + ((skTyk - ykTD1yk - ykTyk) / trYk2) * Yk;
        }
        break;
        case EBroydenJacobianEstimate::DiagonalCauchySchwarz: {
            // Accumulate lumped diagonal inverse Jacobian
            broyden.Fknorm2 += broyden.Fk.col(dkl).squaredNorm();
            broyden.Bknorm2 += (broyden.Xk.col(dkl).array() -
                                broyden.Gkm.col(dkl).array() * broyden.Fk.col(dkl).array())
                                   .square()
                                   .sum();
            auto ddkl = common::Modulo(dkl - 1, broyden.m);
            broyden.FkRowNorm2.col(dkl) =
                broyden.FkRowNorm2.col(ddkl) + broyden.Fk.col(dkl).cwiseSquare();
            Scalar sigma = (broyden.betaF / broyden.sqrtBetaB) *
                           (std::sqrt(broyden.Bknorm2) / broyden.Fknorm2);
            broyden.Gkm.col(dkl) += sigma * broyden.FkRowNorm2.col(dkl).cwiseSqrt();
        }
        default: break;
    }
    ++broyden.k;
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Solve(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, BroydenParams& broyden)
{
    InitializeSolve<TElasticEnergy>(fem, params, broyden);
    for (; broyden.k < params.nMaxIters;)
        Iterate<TElasticEnergy>(fem, params, broyden);
}

template <physics::CHyperElasticEnergy TElasticEnergy>
void Integrate(FemElastoDynamics<TElasticEnergy>& fem, Params const& params, BroydenParams& broyden)
{
    fem.SetupTimeIntegrationOptimization();
    Solve<TElasticEnergy>(fem, params, broyden);
    BackSubstituteIntegratedPositionsIntoVelocities<TElasticEnergy>(fem, params);
    fem.Step();
}

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_BROYDEN_H
