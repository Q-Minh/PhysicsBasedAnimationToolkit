/**
 * @file Enums.h
 * @author Quoc-Minh Ton-That (tonthat.quocminh@gmail.com)
 * @brief Enums for VBD API.
 * @version 0.1
 * @date 2025-10-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef PBAT_SIM_ALGORITHM_VBD_ENUMS_H
#define PBAT_SIM_ALGORITHM_VBD_ENUMS_H

namespace pbat::sim::algorithm::vbd {

/**
 * @brief Vertex integration linear solvers
 */
enum class EVertexIntegrationLinearSolver {
    Inverse, ///< Compute explicit 3x3 inverse
    LLT,     ///< Cholesky factorize
    QR,      ///< MGS QR factorization
    EVD      ///< Eigenvalue decomposition
};

/**
 * @brief Solver for the Broyden least-squares problem
 */
enum class EBroydenLeastSquaresSolver {
    QR,                    ///< QR decomposition
    COD,                   ///< Complete orthogonal decomposition
    LSCG,                  ///< Conjugate gradient
    OneStepSteepestDescent ///< One-step steepest descent
};

/**
 * @brief Broyden Jacobian estimate strategies
 */
// clang-format off
enum class EBroydenJacobianEstimate {
    Identity, ///< Initial Jacobian is identity matrix
    ScaledIdentity, ///< See \cite oren1974SelfScaling
    QuasiCauchyRelationDiagonalUpdating, ///< See \cite zhu1999quasi
    UsdDiagonal, ///< See \cite marjugi2013diagonal
    DiagonalCauchySchwarz ///< Ours
};
// clang-format on

} // namespace pbat::sim::algorithm::vbd

#endif // PBAT_SIM_ALGORITHM_VBD_ENUMS_H
